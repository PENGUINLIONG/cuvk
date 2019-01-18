from os import environ
if "L_BENCH_TYPE" not in environ:
    print("NOTE: benchmark type is not given, CUVK is used by default")
    from cuvk import *
bench_type = environ["L_BENCH_TYPE"]
if bench_type == "CUVK":
    from cuvk import *
elif bench_type == "CUBL":
    from cubl import *
else:
    print("unknown bench type")
    exit()
import matplotlib.pyplot as plot
import numpy as np

if __name__ == '__main__':

    init()

    # Number of deform specs.
    SPEC_COUNT = 100
    # Number of bacteria to be deformed.
    if "L_BAC_COUNT" in environ:
        BAC_COUNT = int(environ["L_BAC_COUNT"])
    else:
        BAC_COUNT = 100
    # Number of universes at first, without deformation.
    INIT_UNIV_COUNT = 20
    # Number of universes to be simulated after deformation.
    UNIV_COUNT = INIT_UNIV_COUNT * SPEC_COUNT
    # Width of each universe.
    UNIV_WIDTH = 360
    # Height of each universe.
    UNIV_HEIGHT = 240

    # Enable this to print deformed cell profiles and to show rendered
    # universes.
    SHOW_DETAIL = False


    # Fill out the memory requirements and create our context, where CUVK
    # resources are allocated.
    mem_req = MemoryRequirements()
    mem_req.nspec = SPEC_COUNT
    mem_req.nbac = BAC_COUNT
    mem_req.nuniv = UNIV_COUNT
    mem_req.width = UNIV_WIDTH
    mem_req.height = UNIV_HEIGHT

    ctxt = Context(0, mem_req)

    # Make up some 'random' deform specs and bacteria.
    deform_specs = []
    bacs = []

    for i in range(SPEC_COUNT):
        deform_spec = DeformSpecs()
        deform_spec.trans_x = 0.7*(i%5)/5 - 0.35
        deform_spec.trans_y = 0.9*(1%5)/5 - 0.45
        deform_spec.stretch_length = 1 + 0.3*((i + 1)/60)
        deform_spec.stretch_width = 1 + 0.3*((i + 1)/60)
        deform_spec.rotate = 3.1415926*(i/60)
        deform_specs.append(deform_spec)

    for i in range(BAC_COUNT):
        bac = Bacterium()
        bac.length = 0.08
        bac.width = 0.03
        bac.x = 0.15*(i%5) + 0.25*(i%2)
        bac.y = 0.15*(i%5) + 0.25*(i%2)
        bac.orient = 3.1415926 * 4 * (i / 60)
        bac.univ = int(i / 10)
        bacs.append(bac)

    # Make up a real universe of nothing.
    real_univ = [0] * UNIV_HEIGHT * UNIV_WIDTH

    # Dispatch deformation tasks on GPU.
    deform_tasks = [ctxt.deform(deform_specs, bacs, 0, INIT_UNIV_COUNT) for i in range(10)]

    # NOTE: If you want to test if the subtraction is actually taken place, make
    # up the real universe like this:
    # ```
    # real_univ = [0.5] * UNIV_HEIGHT * UNIV_WIDTH
    # ```
    # In this way, the sum of the abs(difference) will be half of the total
    # pixel number in the universe.

    # Dispatch evaluation tasks on GPU.
    eval_tasks = []
    for i in range(10):
        task = deform_tasks[i]
        task.busy_wait()
        deformed_bacs = task.result()
        eval_tasks.append(ctxt.eval(deformed_bacs, UNIV_WIDTH, UNIV_HEIGHT, real_univ, 0, UNIV_COUNT))

    eval_tasks[4].busy_wait()
    deinit()
