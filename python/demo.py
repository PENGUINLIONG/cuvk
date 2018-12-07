from cuvk import *
import matplotlib.pyplot as plot
import numpy as np

init()

# Number of deform specs.
SPEC_COUNT = 60
# Number of bacteria to be deformed.
BAC_COUNT = 60
# Number of universes at first, without deformation.
INIT_UNIV_COUNT = 20
# Number of universes to be simulated after deformation.
UNIV_COUNT = INIT_UNIV_COUNT * SPEC_COUNT
# Width of each universe.
UNIV_WIDTH = 64
# Height of each universe.
UNIV_HEIGHT = 64

# Fill out the memory requirements and create our context, where CUVK resources
# are allocated.
mem_req = MemoryRequirements()
mem_req.nspec = SPEC_COUNT
mem_req.nbac = BAC_COUNT
mem_req.nuniv = UNIV_COUNT
mem_req.width = UNIV_WIDTH
mem_req.height = UNIV_HEIGHT
mem_req.share_bac = True

ctxt = Context(0, mem_req)

# Make up some 'random' deform specs and bacteria.
deform_specs = []
bacs = []

for i in range(60):
    deform_spec = DeformSpecs()
    deform_spec.trans_x = 0.7*(i%5)/5 - 0.35
    deform_spec.trans_y = 0.9*(1%3)/5 - 0.45
    deform_spec.stretch_length = 1 + 0.3*((i + 1)/60)
    deform_spec.stretch_width = 1 + 0.3*((i + 1)/60)
    deform_spec.rotate = 3.1415926*(i/60)
    deform_specs.append(deform_spec)

    bac = Bacterium()
    bac.length = 0.08
    bac.width = 0.03
    bac.x = 0.2*(i%2)
    bac.y = 0.2*(i%3)
    bac.orient = 3.1415926 * 4 * (i / 60)
    bac.univ = int(i / 3)
    bacs.append(bac)

# Dispatch deformation tasks on GPU.
task = ctxt.deform(deform_specs, bacs, 0, INIT_UNIV_COUNT)
task.busy_wait()

deformed_bacs = task.result()

# Make up a real universe of nothing.
real_univ = [0] * UNIV_HEIGHT * UNIV_WIDTH

# NOTE: If you want to test if the subtraction is actually taken place, make up
# the real universe like this:
# ```
# real_univ = [0.5] * UNIV_HEIGHT * UNIV_WIDTH
# ```
# In this way, the sum of the abs(difference) will be half of the total pixel
# number in the universe.

# Dispatch evaluation tasks on GPU.
task = ctxt.eval(deformed_bacs, real_univ, 0, UNIV_COUNT)
task.busy_wait()

(sim_univs, costs) = task.result()

# Wrap up data as numpy ndarray.
univs = np.asarray(sim_univs)
univs = univs.reshape((UNIV_COUNT, UNIV_HEIGHT, UNIV_WIDTH))

# Present each image.
for i in range(UNIV_COUNT):
    print("universe #%d, cost=%f, sum(pixels)=%f" %
          (i, costs[i], univs[i].sum()))
    plot.imshow(univs[i], cmap="gray")
    plot.show()

deinit()

