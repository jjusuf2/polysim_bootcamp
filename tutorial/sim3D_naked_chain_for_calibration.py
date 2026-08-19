# Here we run a 3D polymer simulation with no loop extrusion and no sticky monomers

# This will be used to calibrate the length and time scales of the polymer simulation
# by comparing the simulated MSD curve against the experimental MSD of Fbn2 ∆RAD21 
# (see tutorial/calibration.ipynb)

import os, sys
import numpy as np

from polysim import sim3d

outpath = '<path>/<to>/<your>/<folder>'  # the folder in which to write outputs (make sure there is at least a few hundreds of GB free!)


params = sim3d.SimParams(

    npoly = 70000,  # must use same value as main simulation (see sim3D_complete.py)
    density = 0.3,

    ctcf_left  = None,  # no CTCF sites
    ctcf_right = None,                      
    stall = None,
    
    sep = None,  # this turns off loop extrusion (no LEFs)

    monomer_types=None,  # this turns off sticky interactions (no E/P)
    interaction_matrix=None,

    # --- integration ---
    platform = "CUDA",
    gpu = "0",                  # the two GPUs on each computer are called "0" or "1" (use nvidia-smi to check usage)
    integrator = "langevin",    # must use same value as main simulation
    dt = 40,                    # must use same value as main simulation
    colrate = 0.01,             # must use same value as main simulation
    poly_steps_per_block = 40,  # this value is more or less arbitrary for now, since we will find the proper value via calibrating the
                                # MSD of this simulation to an experimental MSD with ∆RAD21

    # --- schedule ---
    initsteps=0,                # LEF-only steps before the polymer starts moving (not needed, because there are no LEFs)
    saveevery=500,              # number of blocks between saves (probably several seconds?); must divide blocks_per_updater, whose default value is 1000
    numsave=36000,              # total number of save-blocks (probably several tens of hours?)
    blocks_per_updater = 1000,  # number of blocks between restarting the smcBondUpdater (no need to change)
    
    # --- output ---
    max_data_length = 100,  # how many blocks to write to each output file (reduce it if you want to see your output faster)
    outpath = outpath,
    flag = "",              # label appended to the auto-generated folder name
)

print(params.summary())

sim3d.run(params)  # run the simulation