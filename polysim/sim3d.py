"""3D polymer + loop-extrusion simulations with polychrom/OpenMM.

Importable module -- nothing runs at import time. Drive it from ``run_sim3D.ipynb``
(or any script) like this::

    from polysim.sim3d import SimParams, run

    params = SimParams(npoly=10000, nchr=1, density=0.2, life=3000, sep=480,
                       ctcf_left=[1000, 3000], ctcf_right=[2000, 4000])
    folder = run(params)

Layout of the package:

    polysim/extrusion.py     CTCF stall arrays + the 1D LEF translocator (no OpenMM)
    polysim/bond_updater.py  pushes LEF positions into OpenMM harmonic bonds
    polysim/sim3d.py         this file: SimParams, polymer/force setup, the run loop

Output goes to ``polysim.OUTPUTS`` by default, which is outside the repository.

Compartments are deliberately not modelled here. Instead of hard-coded A/B/C block
lists you pass ``monomer_types`` and ``interaction_matrix`` straight through to
polychrom's ``heteropolymer_SSW`` if you want type-specific interactions at all; leave
both as None (the default) for a plain homopolymer with soft repulsion.
"""

import os
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np

from polychrom import forcekits, forces, simulation
from polychrom.hdf5_format import HDF5Reporter
from polychrom.starting_conformations import grow_cubic

from . import extrusion
from . import OUTPUTS
from .bond_updater import smcBondUpdater


@dataclass
class SimParams:
    """Everything a run needs. Every field is a plain keyword argument with a default.

    Defaults follow Gabriele et al. 2022 (extrusion) and Goel et al. 2025 (integration),
    as in the original sim3D.py script.
    """

    # ---- polymer ------------------------------------------------------------------
    npoly: int = 40080  # total monomers; ignored if chr_sizes is given
    nchr: int = 1  # number of chains; ignored if chr_sizes is given
    chr_sizes: Optional[Sequence[int]] = None  # explicit per-chain lengths
    density: float = 0.2  # monomers per unit volume inside the confinement
    pbc: bool = False  # periodic box instead of spherical confinement
    confinement_k: float = 1.0  # stiffness of the spherical confinement
    repel: float = 3.0  # nonbonded repulsion energy, kT
    ignore_adjacent: bool = True  # skip nonbonded forces between bonded neighbours
    bond_wiggle: float = 0.1  # backbone bond wiggle distance
    angle_k: float = 1.5  # backbone bending stiffness

    # ---- monomer types (optional; None -> homopolymer) ------------------------------
    monomer_types: Optional[Sequence[int]] = None  # length npoly, values 0..ntypes-1
    interaction_matrix: Optional[Sequence[Sequence[float]]] = None  # symmetric ntypes x ntypes
    attraction_energy: float = 0.0  # background attraction for all pairs, kT

    # ---- loop extrusion -------------------------------------------------------------
    life: float = 3000.0  # LEF lifetime in LEF timesteps
    sep: float = 480.0  # monomers per LEF -> n_lefs = npoly // sep
    vlef: float = 0.05  # p(step per leg per timestep)
    stall: float = 0.8  # CTCF stall probability per encounter
    stallall: bool = False  # stall everywhere (ignores the site lists)
    lifebooststalled: float = 4.0  # lifetime multiplier while stalled at CTCF
    ctcf_left: Optional[Sequence[int]] = None  # sites blocking the left-moving leg
    ctcf_right: Optional[Sequence[int]] = None  # sites blocking the right-moving leg
    smc_bond_wiggle: float = 0.1
    smc_bond_dist: float = 0.5

    # ---- integration ------------------------------------------------------------------
    platform: str = "CUDA"
    gpu: str = "0"
    integrator: str = "langevin"
    dt: int = 40  # timestep, fs
    thermostat: float = 0.01  # collision rate during production
    thermostat0: float = 0.01  # collision rate during equilibration (>= thermostat)
    polysteps: int = 450  # polymer timesteps per LEF timestep
    max_ek: float = 20.0

    # ---- schedule ---------------------------------------------------------------------
    numsave: int = 10000  # saved blocks
    saveevery: int = 100  # LEF steps between saves; must divide 1000
    initskip: int = 80  # saved blocks discarded as equilibration
    initsteps: int = 1000000  # LEF-only steps before the polymer starts
    blocks_per_updater: int = 1000  # bondUpdater is rebuilt this often
    smc_steps_per_block: int = 1

    # ---- io -----------------------------------------------------------------------------
    outpath: str = str(OUTPUTS)  # polysim.OUTPUTS -- deliberately outside the repository
    flag: str = ""  # label appended to the output folder name
    restart_file: str = ""  # path to a pickled conformation to restart from
    save_smc_bonds: bool = True  # dump SMC*.dat + bondsAdded.txt alongside the trajectory

    def __post_init__(self):
        if self.chr_sizes is None:
            base = int(self.npoly) // int(self.nchr)
            sizes = [base] * int(self.nchr)
            sizes[-1] += int(self.npoly) - base * int(self.nchr)  # remainder onto the last chain
            self.chr_sizes = sizes
        else:
            self.chr_sizes = [int(c) for c in self.chr_sizes]
            self.nchr = len(self.chr_sizes)
            self.npoly = int(sum(self.chr_sizes))

        if min(self.chr_sizes) < 2:
            raise ValueError("every chain needs at least 2 monomers, got {0}".format(self.chr_sizes))
        if self.thermostat0 < self.thermostat:
            self.thermostat0 = self.thermostat
        if self.lifebooststalled <= 0:
            raise ValueError("lifebooststalled must be positive")
        if self.blocks_per_updater % self.saveevery != 0:
            raise ValueError("saveevery must divide blocks_per_updater ({0})".format(self.blocks_per_updater))
        if (self.monomer_types is None) != (self.interaction_matrix is None):
            raise ValueError(
                "monomer_types and interaction_matrix must be given together "
                "(leave both None for a homopolymer)"
            )

    # --- derived quantities ---------------------------------------------------------
    @property
    def n_lefs(self):
        return int(self.npoly // self.sep)

    @property
    def chains(self):
        """(start, end, isRing) triples for polychrom's polymer_chains forcekit."""
        edges = np.concatenate([[0], np.cumsum(self.chr_sizes)])
        return [(int(edges[i]), int(edges[i + 1]), False) for i in range(self.nchr)]

    @property
    def confinement_radius(self):
        return (self.npoly / self.density) ** (1.0 / 3.0)

    @property
    def pbc_box(self):
        if not self.pbc:
            return False
        side = (self.npoly / self.density) ** (1.0 / 3.0)
        return [side] * 3

    def to_dict(self):
        d = asdict(self)
        # arrays are not picklable-friendly as dataclass fields; normalise to lists
        for key in ("monomer_types", "interaction_matrix", "ctcf_left", "ctcf_right", "chr_sizes"):
            if isinstance(d[key], np.ndarray):
                d[key] = d[key].tolist()
        return d

    def summary(self):
        """Human-readable rundown of what this configuration will actually do."""
        left, right = self.stall_arrays()
        n_sites = int(((left > 0) | (right > 0)).sum())
        sched = self.schedule()
        lines = [
            "polymer      {0} monomers in {1} chain(s) {2}".format(self.npoly, self.nchr, self.chr_sizes),
            "confinement  {0} r={1:.1f} at density {2}".format(
                "PBC box" if self.pbc else "sphere", self.confinement_radius, self.density
            ),
            "interactions {0}".format(
                "homopolymer (soft repulsion, {0} kT)".format(self.repel)
                if self.interaction_matrix is None
                else "heteropolymer_SSW with {0} types".format(len(np.unique(self.monomer_types)))
            ),
            "LEFs         {0} (1 per {1} monomers), lifetime {2:g}, v={3:g}/step".format(
                self.n_lefs, self.sep, self.life, self.vlef
            ),
            "CTCF         {0} stall site(s), p={1:g} per encounter, lifetime x{2:g} while stalled".format(
                n_sites, self.stall, self.lifebooststalled
            ),
            "schedule     {0} blocks written ({1} of them equilibration, drop those),"
            " {2} LEF steps each, {3} polymer steps per LEF step".format(
                sched["n_blocks"], sched["n_equil_blocks"], sched["save_every"], self.polysteps
            ),
        ]
        return "\n".join(lines)

    # --- setup helpers ----------------------------------------------------------------
    def stall_arrays(self):
        return extrusion.build_stall_arrays(
            self.npoly,
            ctcf_left=self.ctcf_left,
            ctcf_right=self.ctcf_right,
            stall_prob=self.stall,
            stall_all=self.stallall,
        )

    def schedule(self):
        """Resolve the equilibration/production block schedule."""
        save_every = self.saveevery
        skip = int(self.initskip)
        restarting = len(str(self.restart_file)) > 0
        if restarting:
            skip = 0
            save_every = 10  # a restart wants dense sampling right away
        elif skip > 0:
            # make sure equilibration is at least one LEF lifetime long
            while save_every * skip * self.smc_steps_per_block <= self.life:
                skip *= 2

        per_updater = self.blocks_per_updater // save_every
        if (skip * save_every) % self.blocks_per_updater != 0:
            raise ValueError("initskip * saveevery must be a multiple of blocks_per_updater")
        if (self.numsave * save_every) % self.blocks_per_updater != 0:
            raise ValueError("numsave * saveevery must be a multiple of blocks_per_updater")
        if self.numsave * save_every * self.smc_steps_per_block <= self.life:
            raise ValueError("the run is shorter than one LEF lifetime; raise numsave or saveevery")

        updater_skip = save_every * skip // self.blocks_per_updater
        updater_total = (self.numsave + skip) * save_every // self.blocks_per_updater
        return {
            "save_every": save_every,
            "skip_blocks": skip,
            "updater_skip": updater_skip,
            "updater_total": updater_total,
            "saves_per_updater": per_updater,
            # every block is written to the trajectory, including the equilibration ones
            # (they just run at thermostat0); n_equil_blocks tells you how many to drop
            "n_blocks": updater_total * per_updater,
            "n_equil_blocks": updater_skip * per_updater,
            "restarting": restarting,
        }


def make_folder(params, folder=None):
    """Create (and return) the output directory, appending 0001/0002/... to avoid clobbering."""
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        return folder

    bits = [
        "npoly{0}".format(params.npoly),
        "nchr{0}".format(params.nchr),
        "dens{0:g}".format(params.density),
        "life{0:g}".format(params.life),
        "sep{0:g}".format(params.sep),
        "vlef{0:g}".format(params.vlef),
        "dt{0}".format(params.dt),
    ]
    left, right = params.stall_arrays()
    if (left > 0).any() or (right > 0).any():
        bits.append("stallall{0:g}".format(params.stall) if params.stallall else "stallsites{0:g}".format(params.stall))
        if params.lifebooststalled != 1.0:
            bits.append("lifeboost{0:g}".format(params.lifebooststalled))
    if params.interaction_matrix is not None:
        bits.append("ntypes{0}".format(len(np.unique(params.monomer_types))))
    if params.pbc:
        bits.append("PBC")
    if params.integrator != "langevin":
        bits.append(str(params.integrator))
    if len(str(params.restart_file)) > 0:
        bits.append("restart")
    if len(params.flag) > 0:
        bits.append(params.flag)

    ind = 1
    while True:
        candidate = os.path.join(params.outpath, "trajectory{0:04d}_".format(ind) + "_".join(bits))
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        ind += 1


def initial_conformation(params):
    """Starting conformation: a self-avoiding cubic walk, as in the polychrom example.

    The box is chosen slightly denser than the target so the chain starts inside the
    spherical confinement rather than being squeezed into it.
    """
    if len(str(params.restart_file)) > 0:
        with open(params.restart_file, "rb") as f:
            polymer = pickle.load(f)
        polymer = np.asarray(polymer)
        if polymer.shape != (params.npoly, 3):
            raise ValueError(
                "restart file has shape {0}, expected ({1}, 3)".format(polymer.shape, params.npoly)
            )
        return polymer

    box = int((params.npoly / (params.density * 1.2)) ** (1.0 / 3.0))
    return grow_cubic(params.npoly, box)


def nonbonded_force(params):
    """(force_func, kwargs) for polymer_chains -- homopolymer unless types were given."""
    if params.interaction_matrix is None:
        return forces.polynomial_repulsive, {"trunc": params.repel}

    types = np.asarray(params.monomer_types, dtype=int)
    matrix = np.asarray(params.interaction_matrix, dtype=float)
    if types.shape != (params.npoly,):
        raise ValueError("monomer_types must have length npoly={0}, got {1}".format(params.npoly, types.shape))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("interaction_matrix must be square, got shape {0}".format(matrix.shape))
    if not np.allclose(matrix, matrix.T):
        raise ValueError("interaction_matrix must be symmetric")
    if types.min() < 0 or types.max() >= matrix.shape[0]:
        raise ValueError(
            "monomer_types values must index interaction_matrix (0..{0}), got range {1}..{2}".format(
                matrix.shape[0] - 1, types.min(), types.max()
            )
        )

    return forces.heteropolymer_SSW, {
        "interactionMatrix": matrix,
        "monomerTypes": types,
        "extraHardParticlesIdxs": [],
        "repulsionEnergy": params.repel,
        "attractionEnergy": params.attraction_energy,
    }


def build_simulation(params, polymer, reporter, collision_rate):
    """One polychrom Simulation with the polymer loaded and all non-SMC forces added."""
    sim = simulation.Simulation(
        platform=params.platform,
        GPU=params.gpu,
        integrator=params.integrator,
        collision_rate=collision_rate,
        timestep=params.dt,
        max_Ek=params.max_ek,
        N=params.npoly,
        PBCbox=params.pbc_box,
        save_decimals=3,
        reporters=[reporter],
    )
    sim.set_data(polymer, center=True)

    if not params.pbc:
        sim.add_force(forces.spherical_confinement(sim, density=params.density, k=params.confinement_k))

    force_func, force_kwargs = nonbonded_force(params)
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=params.chains,
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": params.bond_wiggle},
            angle_force_func=forces.angle_force,
            angle_force_kwargs={"k": params.angle_k},
            nonbonded_force_func=force_func,
            nonbonded_force_kwargs=force_kwargs,
            except_bonds=params.ignore_adjacent,
        )
    )
    return sim


def make_translocator(params):
    """LEF translocator plus the per-monomer arrays it was built from."""
    stall_left, stall_right = params.stall_arrays()
    arrays = extrusion.build_lef_arrays(
        params.npoly,
        lifetime=params.life,
        vlef=params.vlef,
        stall_left=stall_left,
        stall_right=stall_right,
        life_boost_stalled=params.lifebooststalled,
    )
    return extrusion.make_translocator(arrays, params.n_lefs), arrays


def run(params, folder=None, verbose=True):
    """Run the simulation. Returns the output folder.

    Writes into ``folder`` (auto-named under ``params.outpath`` if not given):
        blocks_*.h5     polychrom trajectory (read with polychrom.hdf5_format.list_URIs)
        paramsDict.pkl  the SimParams as a dict
        sites.npz       the per-monomer LEF arrays actually used
        SMC*.dat        LEF bond lists, if save_smc_bonds
    """
    sched = params.schedule()
    folder = make_folder(params, folder)
    if verbose:
        print(params.summary())
        print("\noutput -> {0}".format(folder))

    with open(os.path.join(folder, "paramsDict.pkl"), "wb") as f:
        pickle.dump(params.to_dict(), f)

    smc_tran, lef_arrays = make_translocator(params)
    np.savez(os.path.join(folder, "sites.npz"), **lef_arrays)

    if params.initsteps > 0:
        if verbose:
            print("equilibrating LEF dynamics for {0} steps...".format(params.initsteps), flush=True)
        tstart = time.time()
        smc_tran.steps(int(params.initsteps))
        if verbose:
            print("  done in {0:.1f} s".format(time.time() - tstart))

    bond_updater = smcBondUpdater(smc_tran)
    polymer = initial_conformation(params)
    reporter = HDF5Reporter(folder=folder, max_data_length=100, overwrite=True)

    kbond_wiggle = params.smc_bond_wiggle
    save_every = sched["save_every"]
    prev_step = 0
    tot_bonds_added = 0
    lef_steps_taken = 0
    cur_bonds = []

    try:
        for updater_count in range(sched["updater_total"]):
            do_save = updater_count >= sched["updater_skip"]
            if verbose:
                print("updater init {0} / {1}{2}".format(
                    updater_count, sched["updater_total"], "" if do_save else "  (equilibration)"), flush=True)

            collision_rate = params.thermostat if do_save else params.thermostat0
            sim = build_simulation(params, polymer, reporter, collision_rate)

            kbond = sim.kbondScalingFactor / (kbond_wiggle ** 2)
            bond_dist = params.smc_bond_dist * sim.length_scale
            bond_updater.setParams({"length": bond_dist, "k": kbond}, {"length": bond_dist, "k": 0})
            cur_bonds, _ = bond_updater.setup(
                bondForce=sim.force_dict["harmonic_bonds"],
                blocks=params.blocks_per_updater,
                smcStepsPerBlock=params.smc_steps_per_block,
            )

            if updater_count in (0, sched["updater_skip"]):
                sim.local_energy_minimization()
            else:
                sim._apply_forces()

            sim.step = prev_step
            for i in range(params.blocks_per_updater):
                if i % save_every == (save_every - 1):
                    sim.do_block(steps=params.polysteps)
                    if params.save_smc_bonds and (i % (10 * save_every) == (10 * save_every - 1)):
                        with open(os.path.join(folder, "SMC{0}.dat".format(sim.step)), "wb") as f:
                            pickle.dump(cur_bonds, f)
                        with open(os.path.join(folder, "bondsAdded.txt"), "a") as f:
                            f.write("{0} {1}\n".format(lef_steps_taken, tot_bonds_added))
                        tot_bonds_added = 0
                        lef_steps_taken = 0
                else:
                    # skip the GPU->host copy for blocks we are not saving
                    sim.integrator.step(params.polysteps)
                    sim.step += params.polysteps
                if i < params.blocks_per_updater - 1:
                    cur_bonds, _, n_added = bond_updater.step(sim.context, countSteps=True)
                    tot_bonds_added += n_added
                    lef_steps_taken += 1

            polymer = sim.get_data()
            prev_step = sim.step
            del sim
            reporter.blocks_only = True
            time.sleep(0.2)
    finally:
        reporter.dump_data()

    if verbose:
        print("\ndone -> {0}".format(folder))
    return folder
