from ctypes import *

LIBCUVK = windll.LoadLibrary("./build/Debug/libcuvk.dll")

CUVK = None

class CuvkInstance:
    def __init__(self):
        if not LIBCUVK.cuvkInitialize(True):
            raise RuntimeError("Unable to initialize CUVK.")
    def __del__(self):
        LIBCUVK.cuvkDeinitialize()



def init():
    global CUVK
    CUVK = CuvkInstance()

def deinit():
    global CUVK
    CUVK = None

class DeformSpecs(Structure):
    _fields_ = [('trans_x', c_float),
                ('trans_y', c_float),
                ('stretch_length', c_float),
                ('stretch_width', c_float),
                ('rotate', c_float),
                ('_pad0', c_int)]

    def __repr__(self):
        return "[translate=(%f, %f), stretch=(%f,%f), rotate=%f]" % (self.translate_x, self.translate_y, self.stretch_length, self.stretch_width, self.rotate)

class Bacterium(Structure):
    _fields_ = [('x', c_float),
                ('y', c_float),
                ('length', c_float),
                ('width', c_float),
                ('orient', c_float),
                ('univ', c_uint)]

    def __repr__(self):
        return "[position=(%f, %f), size=(%f,%f), orientation=%f, universe=%d]" % (self.x, self.y, self.length, self.width, self.orient, self.univ)

class MemoryRequirements(Structure):
    _fields_ = [('nspec', c_uint),
                ('nbac', c_uint),
                ('nuniv', c_uint),
                ('width', c_uint),
                ('height', c_uint),
                ('share_bac', c_uint)]

class DeformationInvocation(Structure):
    _fields_ = [('deform_specs', POINTER(DeformSpecs)),
                ('nspec', c_uint),
                ('bacs', POINTER(Bacterium)),
                ('nbac', c_uint),
                ('base_univ', c_uint),
                ('nuniv', c_uint),
                ('bacs_out', POINTER(Bacterium))]
    def __init__(self, specs, bacs, base_univ, nuniv):
        self.nspec = len(specs)
        self.specs_buf = (DeformSpecs * self.nspec)()
        for i in range(self.nspec):
            self.specs_buf[i] = specs[i]
        self.deform_specs = cast(self.specs_buf, POINTER(DeformSpecs))

        self.nbac = len(bacs)
        self.bacs_buf = (Bacterium * self.nbac)()
        for i in range(self.nspec):
            self.bacs_buf[i] = bacs[i]
        self.bacs = cast(self.bacs_buf, POINTER(Bacterium))

        self.base_univ = c_uint(base_univ)
        self.nuniv = c_uint(nuniv)

        self.bacs_out_buf = (Bacterium * (self.nbac * self.nspec))()
        self.bacs_out = cast(self.bacs_out_buf, POINTER(Bacterium))

class EvaluationInvocation(Structure):
    _fields_ = [('bacs', POINTER(Bacterium)),
                ('nbac', c_uint),
                ('sim_univs', POINTER(c_float)),
                ('real_univ', POINTER(c_float)),
                ('nsim_univ', c_uint),
                ('base_sim_univ', c_uint),
                ('costs', POINTER(c_float))]
    def __init__(self, bacs, real_univ, base_sim_univ, nsim_univ):
        self.nbac = len(bacs)
        self.bacs_buf = (Bacterium * self.nbac)()
        for i in range(self.nbac):
            self.bacs_buf[i] = bacs[i]
        self.bacs = cast(self.bacs_buf, POINTER(Bacterium))

        univ_size = len(real_univ)

        self.real_univ_buf = (c_float * (univ_size))()
        for i in range(univ_size):
            self.real_univ_buf[i] = real_univ[i]
        self.real_univ = cast(self.real_univ_buf, POINTER(c_float))

        self.base_sim_univ = base_sim_univ
        self.nsim_univ = nsim_univ

        self.sim_univs_buf = (c_float * (nsim_univ * univ_size))()
        self.sim_univs = cast(self.sim_univs_buf, POINTER(c_float))
        self.costs_buf = (c_float * nsim_univ)()
        self.costs = cast(self.costs_buf, POINTER(c_float))

def enumerate_physical_devices():
    size = c_int()
    LIBCUVK.cuvkEnumeratePhysicalDevices(byref(size), 0)
    phys_dev = create_string_buffer(b'\000' * size)
    LIBCUVK.cuvkEnumeratePhysicalDevices(byref(size), 0)
    return phys_dev.value


class Task:
    NOT_READY = 0
    OK = 1
    ERROR = 2

    def __init__(self, ctxt, handle, invoke):
        self._inst = CUVK
        self._ctxt = ctxt
        self._handle = handle
        self._status = self.NOT_READY
        self._invoke = invoke
    def __del__(self):
        LIBCUVK.cuvkDestroyTask(self._handle)

    def poll(self):
        """
        Poll the task and check its status. Returns the status of the task that
        can be `NOT_READY`, `OK` or `ERROR`.
        """
        if (self._status is self.NOT_READY):
            self._status = LIBCUVK.cuvkPoll(self._handle)
        return self._status
    def busy_wait(self):
        """
        Actively wait for the task to finish by loop-checking the task status.
        """
        while self.poll() is self.NOT_READY:
            pass
        return self._status

class DeformationTask(Task):
    def __init__(self, ctxt, invoke):
        if type(invoke) is not DeformationInvocation:
            raise TypeError("`invoke` is not DeformationInvocation.")
        task = c_void_p()
        if not LIBCUVK.cuvkInvokeDeformation(
            ctxt._handle, byref(invoke), byref(task)):
            raise RuntimeError("Unable to create deformation task.")
        super().__init__(ctxt, task, invoke)

    def result(self):
        """
        Retrieve the result of the deformation task. The return type is a list
        of bacteria.
        """
        if self._status is self.NOT_READY:
            raise RuntimeError("Task result is not ready yet.")
        elif self._status is self.ERROR:
            raise RuntimeError("Error occurred during execution.")
        else:
            return self._invoke.bacs_out_buf

class EvaluationTask(Task):
    def __init__(self, ctxt, invoke):
        if type(invoke) is not EvaluationInvocation:
            raise TypeError("`invoke` is not EvaluationInvocation.")
        task = c_void_p()
        if not LIBCUVK.cuvkInvokeEvaluation(
            ctxt._handle, byref(invoke), byref(task)):
            raise RuntimeError("Unable to create deformation task.")
        super().__init__(ctxt, task, invoke)

    def result(self):
        """
        Retrieve the result of the evaluation task. The type is a 2-tuple of the
        simulated universes and the cost of each universe.
        """
        if self._status is self.NOT_READY:
            raise RuntimeError("Task result is not ready yet.")
        elif self._status is self.ERROR:
            raise RuntimeError("Error occurred during execution.")
        else:
            return (self._invoke.sim_univs_buf, self._invoke.costs_buf)


class Context:
    def __init__(self, phys_dev_idx, mem_req):
        self.inst = CUVK
        ctxt = c_void_p()
        LIBCUVK.cuvkCreateContext(phys_dev_idx, byref(mem_req), byref(ctxt))
        self._handle = ctxt
    def __del__(self):
        LIBCUVK.cuvkDestroyContext(self._handle)

    def deform(self, specs, bacs, base_univ, nuniv):
        """
        Dispatch deformation task. Returns a dispatched deformation task whose
        result is a list of deformed bacteria.
        """
        invoke = DeformationInvocation(specs, bacs, base_univ, nuniv)
        return DeformationTask(self, invoke)

    def eval(self, bacs, real_univ, base_sim_univ, nsim_univ):
        """
        Dispatch evaluation task
        """
        invoke = EvaluationInvocation(bacs, real_univ, base_sim_univ, nsim_univ)
        return EvaluationTask(self, invoke)

