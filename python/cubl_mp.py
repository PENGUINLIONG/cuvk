from cv2 import cv2
import numpy as np
from math import sin, cos, pi
import time
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
# CUBL for CellUniverse Baseline.
#
# This module implement all same APIs as CUVK but barely in Python. Algorithms
# are _adapted_ from the original CellUniverse implementation as an baseline of
# acceleration performance. The deformation task is likely to be less time-
# consuming than the original impl.

POOL = None
def init():
    global POOL
    POOL = Pool(max(cpu_count() // 2, 1))
def deinit():
    global POOL
    POOL.close()
    POOL.join()
    POOL = None

class DeformSpecs:
    def __init__(self):
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.stretch_length = 1.0
        self.stretch_width = 1.0
        self.rotate = 0.0

    def __repr__(self):
        return "[translate=(%f, %f), stretch=(%f,%f), rotate=%f]" % (self.trans_x, self.trans_y, self.stretch_length, self.stretch_width, self.rotate)

class Bacterium:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.length = 1.0
        self.width = 1.0
        self.orient = 0.0
        self.univ = -1
    def __repr__(self):
        return "[position=(%f, %f), size=(%f,%f), orientation=%f, universe=%d]" % (self.x, self.y, self.length, self.width, self.orient, self.univ)

class MemoryRequirements:
    pass

class DeformationInvocation:
    def __init__(self, specs, bacs, base_univ, nuniv):
        self.specs = specs
        self.bacs = bacs
        self.base_univ = base_univ
        self.nuniv = nuniv

class EvaluationInvocation:
    def __init__(self, bacs, width, height, real_univ, base_sim_univ, nsim_univ):
        self.bacs = bacs
        self.width = width
        self.height = height
        self.real_univ = real_univ
        self.base_sim_univ = base_sim_univ
        self.nsim_univ = nsim_univ

class Task:
    NOT_READY = 0
    OK = 1
    ERROR = 2

    def __init__(self, ctxt, proc, invoke):
        self._proc = proc
        self._invoke = invoke

    def poll(self):
        self._result = self._proc(self._invoke)
        return self.OK

    def busy_wait(self):
        return self.poll()

def deform_cell(bac: Bacterium, spec: DeformSpecs, deform_idx, base_univ, nuniv):
    rv = Bacterium()
    rv.x = bac.x + spec.trans_x
    rv.y = bac.y + spec.trans_y
    rv.length = bac.length * spec.stretch_length
    rv.width = bac.width * spec.stretch_width
    rv.orient = bac.orient + spec.rotate
    rv.univ = bac.univ + base_univ + nuniv * deform_idx
    return rv

class DeformationTask(Task):
    def __init__(self, ctxt, invoke):
        super().__init__(ctxt, self.proc, invoke)
    def proc(self, invoke):
        beg = time.clock()
        args = [(bac, spec, idx, invoke.base_univ, invoke.nuniv) for idx, spec in enumerate(invoke.specs) for bac in invoke.bacs]
        rv = POOL.starmap(deform_cell, args)
        end = time.clock()
        print("deformation task takes %f" % ((end - beg) * 1000))
        return rv

    def result(self):
        return self._result


def generate_image_cv2(bac, sim_univs, base_univ, nuniv, univ_width, univ_height):
    univ = sim_univs[bac.univ - base_univ]
    mn = min(univ_width, univ_height)
    length = bac.length * mn / 2
    width = bac.width * mn / 2
    bac_x = (bac.x * mn + univ_width) / 2
    bac_y = (bac.y * mn + univ_height) / 2

    x = bac_x - length*cos(bac.orient)
    y = bac_y + length*sin(bac.orient)
    head_pos = np.array([x, y, 0])

    end_point_1 = np.array([x + width*cos(bac.orient - pi/2),
                            y - width*sin(bac.orient - pi/2), 0])
    end_point_2 = np.array([x - width*cos(bac.orient - pi/2),
                            y + width*sin(bac.orient - pi/2), 0])

    x = bac_x + length*cos(bac.orient)
    y = bac_y - length*sin(bac.orient)
    tail_pos = np.array([x, y, 0])

    end_point_3 = np.array([x - width*cos(bac.orient - pi/2),
                                    y + width*sin(bac.orient - pi/2), 0])
    end_point_4 = np.array([x + width*cos(bac.orient - pi/2),
                                    y - width*sin(bac.orient - pi/2), 0])

    # head and tail
    cv2.circle(univ, tuple(head_pos[:2].astype(int)),
               int(width), 1, -1)
    cv2.circle(univ, tuple(tail_pos[:2].astype(int)),
               int(width), 1, -1)

    # body
    points = [tuple(end_point_1[:2]),
              tuple(end_point_2[:2]),
              tuple(end_point_3[:2]),
              tuple(end_point_4[:2])]
    points = np.array([(int(point[0]), int(point[1])) for point in points])
    cv2.fillConvexPoly(univ, points, 1, 1)

def calc_costs(sim_univ, real_univ):
    return np.sum(sim_univ - real_univ)

class EvaluationTask(Task):
    def __init__(self, ctxt, invoke):
        super().__init__(ctxt, self.proc, invoke)
    def proc(self, invoke):
        beg = time.clock()
        sim_univs = np.zeros((invoke.nsim_univ, invoke.height, invoke.width),
                             dtype=np.float)
        args = [(bac, sim_univs, invoke.base_sim_univ, invoke.nsim_univ, invoke.width, invoke.height) for bac in invoke.bacs]
        POOL.starmap(generate_image_cv2, args)
        args = [(sim_univs[i].flatten(), invoke.real_univ) for i in range(sim_univs.shape[0])]
        costs = POOL.starmap(calc_costs, args)
        end = time.clock()
        print("evaluation task takes %f" % ((end - beg) * 1000))
        return (sim_univs, costs)

    def result(self):
        return self._result

class Context:
    def __init__(self, phys_dev_idx, mem_req):
        self.mem_req = mem_req

    def deform(self, specs, bacs, base_univ, nuniv):
        invoke = DeformationInvocation(specs, bacs, base_univ, nuniv)
        return DeformationTask(self, invoke)

    def eval(self, bacs, width, height, real_univ, base_sim_univ, nsim_univ):
        invoke = EvaluationInvocation(bacs, width, height, real_univ, base_sim_univ, nsim_univ)
        return EvaluationTask(self, invoke)
