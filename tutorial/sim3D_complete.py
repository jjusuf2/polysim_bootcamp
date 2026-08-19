# This Python script runs a 3D polymer simulation of James' fictitious region from Jusuf et al. 2026,
# which has CTCF sites/loop extrusion and sticky monomers

# Note: some parameters differ slightly from the original version published with the paper

import os, sys
import numpy as np

from polysim import sim3d

outpath = '<path>/<to>/<your>/<folder>'  # the folder in which to write outputs (make sure there is at least a few hundreds of GB free!)


## Enter basic chromosome design parameters ##

region_size = 2000  # 2000 kb = 2 Mb
num_regions = 35    # the region repeats 35 times across the entire chromosome
npoly = region_size * num_regions  # total length of chromosome (70,000 monomers = 70 Mb)

## Enter details about CTCF sites ##

ctcf_left_probs  = {200: 0.5,  # RIGHT facing motifs (stalls LEFT leg of cohesin)
                    330: 0.2,  # each line is <monomer_index>: <stall_probability>
                    724: 0.6,
                    1425: 0.25,
                    1433: 0.15,
                    1604: 0.2}
ctcf_right_probs = {574: 0.4,  # LEFT facing motifs (stalls RIGHT leg of cohesin)
                    694: 0.5,  # each line is <monomer_index>: <stall_probability>
                    866: 0.6,
                    1241: 0.05,
                    1390: 0.4,
                    1580: 0.4,
                    1752: 0.5,
                    1800: 0.1}
ctcf_left_probs_all = sim3d.tile_site_probs(ctcf_left_probs, period=region_size, length=npoly)
ctcf_right_probs_all = sim3d.tile_site_probs(ctcf_right_probs, period=region_size, length=npoly)

## Enter details about sticky monomers ##

sticky_monomers = np.array([250, 372, 540, 745, 775, 833, 961, 1202, 1330, 1640, 1722])
sticky_monomers_all = sim3d.tile_sites(sticky_monomers, period=region_size, length=npoly)

monomer_types = np.zeros(npoly, dtype='int')
monomer_types[sticky_monomers_all] = 1

sticky_interaction_energy = 3  # in units of k_B*T
interaction_matrix = np.array([[0, 0],[0, sticky_interaction_energy]])


## Now set up the SimParams object, which contains all the parameters ##

params = sim3d.SimParams(

    # --- CHROMOSOME DESIGN ---
    npoly = npoly,
    density = 0.3,  # number of monomers per unit volume

    #    CTCF sites
    ctcf_left  = ctcf_left_probs_all,
    ctcf_right = ctcf_right_probs_all,
    stall = None,  # usually we would set a global stall probability here, but since we are
                   # assigning each CTCF site a different stall probability using the dicts
                   # above, we set this to None
    
    #    sticky interactions
    monomer_types = monomer_types,
    interaction_matrix = interaction_matrix,

    # --- INTEGRATION ---
    platform = "CUDA",
    gpu = "0",  # the two GPUs on each computer are called "0" or "1" (use nvidia-smi to check usage)
    integrator = "langevin",
    dt = 40,  # this sets how big the polymer timesteps (integration timesteps) are,
                # but it doesn't matter too much, since it is in arbitarary units, and we
                # will save far less often anyway.
                # changing this drastically may affect numerical stability (40 is a good value)
    colrate = 0.01,   # collision rate (0.01 is a good value)
    poly_steps_per_block = 33,  # a "block" is defined as this many polymer timesteps;
                                # the "block" serves as the most fundamental timestep
                                # for loop extrusion and saving.
                                # get the appropriate value of poly_steps_per_block from MSD calibration.
                                # for this parameter set, 33 poly_steps_per_block --> 1 block = 20 ms

    # --- LOOP EXTRUDERS ---
    life = 75000,            # mean LEF lifetime (in blocks)
    sep = 240,               # mean genomic separation between LEFs (in monomers)
    vlef = 0.0025,           # probability that one leg of a LEF takes a step, per block;
                             # this determines the LEF extrusion speed
    lifebooststalled = 4,    # CTCF lifetime gets multiplied by this number when both sides are stalled

    # --- SCHEDULE ---
    initsteps = 540000,         # LEF-only blocks before the polymer starts moving, to equilibrate LEF dynamics
                                # typically want this to be at least a few LEF lifetimes
    saveevery = 50,             # number of blocks between saves; must divide blocks_per_updater, whose default value is 1000
    numsave = 360000,           # total number of saved blocks for which to run the simulation
    blocks_per_updater = 1000,  # number of blocks between restarting the smcBondUpdater (no need to change)

    # --- output ---
    max_data_length = 100,  # how many blocks to write to each output file (reduce it if you want to see your output faster)
    outpath = outpath,      # this was set earlier
    flag = "",              # label appended to the auto-generated folder name
)

print(params.summary())

sim3d.run(params)  # run the simulation