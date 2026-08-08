"""1D loop-extrusion parameter sweep.

Mirrors sweep_1D.ipynb, but runs ``run_1d_sim`` over the full cross product of
vlef x stall x life x lifebooststalled (4 * 4 * 3 * 4 = 192 configurations),
with ``nworkers`` simulations in flight at a time. Every other parameter is held
at the value the notebook uses.

Each configuration writes LEFpositions.npy and sites.npz into its own
subdirectory of outputs/sweep1D, named after the parameters that define it.

Run it with the polysim3 environment::

    /mnt/md0/jjusuf/miniconda3/envs/polysim3/bin/python sweep1D.py
"""

import itertools
import multiprocessing as mp
import os

import numpy as np
from tqdm import tqdm

from polysim import extrusion, OUTPUTS

# --- chain and LEFs ---
npoly = 70000          # monomers
sep = 240              # monomers per LEF -> nlefs = npoly // sep

# --- CTCF ---
ctcf_left = extrusion.tile_sites([200, 330, 724, 1425, 1433, 1604], period=2000, length=npoly)  # right pointing
ctcf_right = extrusion.tile_sites([574, 694, 866, 1241, 1390, 1580, 1752, 1800], period=2000, length=npoly)  # left pointing

# --- schedule ---
initsteps = 2_000_000  # equilibration steps, discarded
numsave = 7200         # recorded frames
saveevery = 3000       # LEF steps between frames

# --- swept parameters ---
vlef_values = [0.00125, 0.0025, 0.005, 0.01]       # p(step per leg per timestep)
stall_values = [0.125, 0.25, 0.5, 0.8]             # stall probability per encounter
life_values = [37000, 75000, 150000]               # LEF lifetime, in LEF timesteps
lifebooststalled_values = [1, 2, 4, 8]             # lifetime multiplier while stalled at a CTCF

# --- sweep execution ---
nworkers = 12
outroot = OUTPUTS / "sweep1D"


def config_name(sep, life, vlef, stall, lifebooststalled, saveevery):
    """Directory name encoding the parameters that distinguish one run from another."""
    return (
        "sep{0:g}_life{1:g}_vlef{2:g}_stall{3:g}"
        "_lifebooststalled{4:g}_saveevery{5:g}"
    ).format(sep, life, vlef, stall, lifebooststalled, saveevery)


def run_1d_sim(npoly, sep, life, vlef, stall, ctcf_left, ctcf_right, lifebooststalled,
               initsteps, numsave, saveevery, outdir):

    nlefs = npoly // sep

    stall_left, stall_right = extrusion.build_stall_arrays(
        npoly, ctcf_left=ctcf_left, ctcf_right=ctcf_right, stall_prob=stall)

    arrays = extrusion.build_lef_arrays(
        npoly, lifetime=life, vlef=vlef,
        stall_left=stall_left, stall_right=stall_right,
        life_boost_stalled=lifebooststalled)

    smc = extrusion.make_translocator(arrays, nlefs)

    smc.steps(initsteps)

    positions = np.zeros((numsave, nlefs, 2), dtype=np.int64)
    for i in range(numsave):
        smc.steps(saveevery)
        left, right = smc.getLEFs()
        positions[i, :, 0] = left
        positions[i, :, 1] = right

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "LEFpositions.npy"), positions)
    np.savez(os.path.join(outdir, "sites.npz"), **arrays)
    return outdir


def run_one(params):
    """Pool worker: unpack one swept combination and run it, reporting failures."""
    vlef, stall, life, lifebooststalled = params
    name = config_name(sep, life, vlef, stall, lifebooststalled, saveevery)
    try:
        run_1d_sim(npoly, sep, life, vlef, stall, ctcf_left, ctcf_right,
                   lifebooststalled, initsteps, numsave, saveevery,
                   os.path.join(outroot, name))
    except Exception as exc:  # keep the rest of the sweep going
        return name, "{0}: {1}".format(type(exc).__name__, exc)
    return name, None


def main():
    configs = list(itertools.product(
        vlef_values, stall_values, life_values, lifebooststalled_values))

    os.makedirs(outroot, exist_ok=True)
    print("{0} configurations, {1} workers, {2} LEFs each".format(
        len(configs), nworkers, npoly // sep))
    print("writing to", outroot)

    failures = []
    with mp.Pool(nworkers) as pool:
        for name, error in tqdm(pool.imap_unordered(run_one, configs),
                                total=len(configs)):
            if error is not None:
                failures.append((name, error))
                tqdm.write("FAILED {0} -- {1}".format(name, error))

    print("done: {0}/{1} succeeded".format(len(configs) - len(failures), len(configs)))
    for name, error in failures:
        print("  failed:", name, "--", error)


if __name__ == "__main__":
    main()
