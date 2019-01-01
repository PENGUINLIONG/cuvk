from cubl import *
#from cuvk import *
import matplotlib.pyplot as plot
import numpy as np

init()

# Number of deform specs.
SPEC_COUNT = 100
# Number of bacteria to be deformed.
BAC_COUNT = 100
# Number of universes at first, without deformation.
INIT_UNIV_COUNT = 20
# Number of universes to be simulated after deformation.
UNIV_COUNT = INIT_UNIV_COUNT * SPEC_COUNT
# Width of each universe.
UNIV_WIDTH = 360
# Height of each universe.
UNIV_HEIGHT = 240

# Enable this to print deformed cell profiles and to show rendered universes.
SHOW_DETAIL = True


# Fill out the memory requirements and create our context, where CUVK resources
# are allocated.
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

# Dispatch deformation tasks on GPU.
task = ctxt.deform(deform_specs, bacs, 0, INIT_UNIV_COUNT)
task.busy_wait()

deformed_bacs = task.result()

print("NOTE: You can stop enumeration by inputing any thing non-empty.")
if SHOW_DETAIL:
    for deformed_bac in deformed_bacs:
        print(deformed_bac, end='')
        if len(input()) != 0:
            break

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
task = ctxt.eval(deformed_bacs, UNIV_WIDTH, UNIV_HEIGHT, real_univ, 0, UNIV_COUNT)
task.busy_wait()

(sim_univs, costs) = task.result()

# Wrap up data as numpy ndarray.
univs = np.asarray(sim_univs)
univs = univs.reshape((UNIV_COUNT, UNIV_HEIGHT, UNIV_WIDTH))

match = True

# Present each image.
print("NOTE: You can stop enumeration by inputing any thing non-empty.")
for i in range(UNIV_COUNT):
    match &= costs[i] == univs[i].sum()
    if SHOW_DETAIL:
        print("universe #%d, cost=%f, sum(pixels)=%f" %
            (i, costs[i], univs[i].sum()), end='')
        plot.imshow(univs[i], cmap="gray")
        plot.show(block=False)
        if len(input()) != 0:
            break

print("cost and white pixel count are%s equal" % "" if match else " not")

deinit()
