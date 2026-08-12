# Here we run a 3D polymer simulation with no loop extrusion

import os, sys
import numpy as np

from polysim import extrusion, OUTPUTS
from polysim.sim3d import SimParams, run

# the first import compiles LEF_Dynamics.pyx via pyximport -- a few seconds, once
print("output goes to", OUTPUTS)

npoly = 70000  # total number of monomers; ignored if chr_sizes is given
region_size = 2000
num_regions = npoly // region_size
region_starts = np.arange(num_regions) * region_size

left_probs  = {200: 0.5, 330: 0.2, 724: 0.6, 1425: 0.25, 1433: 0.15, 1604: 0.2}
right_probs = {574: 0.4, 694: 0.5, 866: 0.6, 1241: 0.05, 1390: 0.4, 1580: 0.4, 1752: 0.5, 1800: 0.1}

monomer_types = np.zeros(npoly, dtype='int')
sticky_elements = np.array([250, 372, 540, 745, 775, 833, 961, 1202, 1330, 1640, 1722])
num_sticky_elements = len(sticky_elements)
sticky_elements_all = np.repeat(region_starts,num_sticky_elements) + np.tile(sticky_elements, num_regions)
monomer_types[sticky_elements_all] = 1
EP_interaction_energy = 3
interaction_matrix = np.array([[0, 0],[0, EP_interaction_energy]])

params = SimParams(
    npoly=npoly,
    density=0.3,  # number of monomers per unit volume

    # --- CTCF sites ---
    # a periodic array of CTCF sites
    # IMPORTANT: LEFT means stalls the left side of a LEF ("right-pointing")
    
    # probability that each motif stalls a passing leg, within one 2000-monomer repeat
    ctcf_left  = extrusion.tile_site_probs(left_probs,  period=2000, length=npoly),
    ctcf_right = extrusion.tile_site_probs(right_probs, period=2000, length=npoly),
                                    
    stall=None,
    stallall=False,  # True = stall everywhere; ignores the lists below
                        # IMPROVE THIS DESCRIPTION

    # loop extrusion
    life=75000,            # LEF lifetime, in LEF timesteps
    sep=240,               # monomers per LEF -> n_lefs = npoly // sep
    vlef=0.0025,           # p(step per leg per timestep)
    lifebooststalled=4,    # lifetime multiplier while stalled at a CTCF

    # no sticky interactions
    monomer_types=monomer_types,
    interaction_matrix=interaction_matrix,

    # --- integration ---
    platform="CUDA",
    gpu="1",
    integrator="langevin",
    dt=40,
    colrate=0.01, colrate0=0.01,
    poly_steps_per_block=40,  # polymer timesteps per block; aim for ~20 ms

    # --- schedule ---
    numsave=360000,        # total number of save-blocks (100 hours)
    saveevery=50,          # number of blocks between saves (1 second); must divide blocks_per_updater, whose default value is 1000
    initskip=0,            # blocks of equilibration for 3D polymer
    initsteps=540000,      # LEF-only steps before the polymer starts moving (3 hours)
    blocks_per_updater = 1000,  # the default

    # --- output ---
    outpath=os.path.join(OUTPUTS),
    flag="",             # label appended to the auto-generated folder name
)

print(params.summary())

run(params)