# Here we run a 3D polymer simulation with no loop extrusion

import os, sys
import numpy as np

from polysim import extrusion, OUTPUTS
from polysim.sim3d import SimParams, run

# the first import compiles LEF_Dynamics.pyx via pyximport -- a few seconds, once
print("output goes to", OUTPUTS)


params = SimParams(
    npoly=70000,  # total number of monomers; ignored if chr_sizes is given
    density=0.3,  # number of monomers per unit volume

    ctcf_left  = None,
    ctcf_right = None,                      
    stall = None,
    
    # no loop extrusion
    sep = None,

    # no sticky interactions
    monomer_types=None,
    interaction_matrix=None,

    # --- integration ---
    platform="CUDA",
    gpu="0",
    integrator="langevin",
    dt=40,
    colrate=0.01, colrate0=0.01,
    poly_steps_per_block=33,  # polymer timesteps per block; aim for ~20 ms

    # --- schedule ---
    numsave=36000,         # total number of save-blocks (100 hours)
    saveevery=500,         # number of blocks between saves (10 seconds); must divide blocks_per_updater, whose default value is 1000
    initskip=0,            # blocks of equilibration for 3D polymer
    initsteps=0,           # LEF-only steps before the polymer starts moving (not needed)
    blocks_per_updater = None,  # no need for bond updater here

    # --- output ---
    outpath=os.path.join(OUTPUTS, 'calibration3D'),
    flag="",             # label appended to the auto-generated folder name
)

print(params.summary())

run(params)
