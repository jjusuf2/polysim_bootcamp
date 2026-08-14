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

Set ``sep=None`` to turn loop extrusion off entirely and run pure polymer dynamics --
no translocator, no LEF bonds, no ``sites.npz`` and no ``SMCs`` in the h5. The block schedule is
untouched, so a ``sep=None`` run samples on exactly the same cadence as the extrusion
run you want to compare it against. Pair it with ``blocks_per_updater=None`` to run each
phase as a single Simulation: the chunking only exists to bound what the bond updater
precalculates, and with no LEFs there is nothing to precalculate.
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
    npoly: int = 70000  # total number of monomers; ignored if chr_sizes is given
    nchr: int = 1  # number of chromosomes (chains); ignored if chr_sizes is given
    chr_sizes: Optional[Sequence[int]] = None  # explicit lengths of each chromosome
    density: float = 0.3  # number of monomers per unit volume
    pbc: bool = False  # periodic box instead of spherical confinement
    confinement_k: float = 1.0  # stiffness of the spherical confinement
    repel: float = 3.0  # nonbonded repulsion energy, in kT, a.k.a. "trunc"
    ignore_adjacent: bool = True  # skip nonbonded forces between bonded neighbors, a.k.a. "except_bonds"
    bond_wiggle: float = 0.1  # backbone bond wiggle distance, in monomer units
    angle_k: float = 0.05  # backbone bending stiffness, in monomer units
                           # we are making a very flexible polymer, basically not necessary here

    # ---- monomer types (optional; None -> homopolymer) ------------------------------
    monomer_types: Optional[Sequence[int]] = None  # an array of types (0,1,2,...) of every monomer in the polymer (length = npoly)
    interaction_matrix: Optional[Sequence[Sequence[float]]] = None  # matrix of attraction energies between the different types, kT
    attraction_energy: float = 0.0  # background attraction energy between all monomers, kT

    # ---- loop extrusion -------------------------------------------------------------
    # the time units here are in LEF timesteps; one block advances the LEF clock by
    # "smc_steps_per_block" and the polymer clock by "poly_steps_per_block" (both below)
    # set sep=None to switch extrusion off entirely; every other field in this block is then unused
    life: float = 3000.0  # LEF lifetime
    sep: Optional[float] = 480  # "separation", defined as # of monomers per LEF -> n_lefs = npoly // sep
                                # None -> no LEFs at all (pure polymer dynamics)
    vlef: float = 0.0025  # 1-sided extrusion motor velocity of LEF, in monomers per timestep
    stall: Optional[float] = 0.6  # probability that a CTCF site will stall a passing LEF
                                  # None -> every site must carry its own probability (dict form below)
    stallall: bool = False  # stall everywhere (ignores the site lists)
    lifebooststalled: float = 4.0  # LEF lifetime multiplier while both sides are stalled CTCFs
    ctcf_left: Optional[Sequence[int]] = None  # array of monomers that are able to block the left-moving LEF leg (i.e., "RIGHT-facing" motifs)
    ctcf_right: Optional[Sequence[int]] = None  # array of monomers that are able to block the right-moving LEF leg (i.e., "LEFT-facing" motifs)
                                               # either list: every site stalls with probability `stall`
                                               # or dict {monomer index: probability}: per-site stall probabilities, and `stall` is unused
    smc_bond_wiggle: float = 0.2
    smc_bond_dist: float = 0.5

    # ---- integration ------------------------------------------------------------------
    platform: str = "CUDA"
    gpu: str = "0"  # the two GPU's in each computer are called "0" and "1"; use nvidia-smi to check usage
    integrator: str = "langevin"
    dt: int = 40  # timestep, fs
    colrate: float = 0.01   # collision rate during production
    colrate0: float = 0.01  # collision rate during equilibration (>= colrate)
    poly_steps_per_block: int = 450  # polymer timesteps per block
    max_ek: float = 20.0

    # ---- schedule ---------------------------------------------------------------------
    # a "block" is one iteration of the inner loop: poly_steps_per_block polymer steps and,
    # with extrusion on, smc_steps_per_block LEF timesteps. Every saveevery-th one is written.
    numsave: int = 5000000  # blocks written to the trajectory
    saveevery: int = 500  # blocks between saves; must divide blocks_per_updater
    initskip: int = 300  # blocks of equilibration, run at colrate0 and (by default) not written
    initsteps: int = 2000000  # LEF-only steps before the polymer starts
    blocks_per_updater: Optional[int] = 1000  # rebuild the sim (and the bondUpdater) this often;
                                              # None -> one chunk per phase, i.e. as few rebuilds as
                                              # the schedule allows (extrusion must be off)
    smc_steps_per_block: int = 1

    # ---- io -----------------------------------------------------------------------------
    outpath: str = str(OUTPUTS)  # polysim.OUTPUTS -- deliberately outside the repository
    flag: str = ""  # label appended to the output folder name
    restart_file: str = ""  # path to a pickled conformation to restart from
    save_smc_bonds: bool = True  # store LEF positions under "SMCs" in each saved h5 block, + bondsAdded.txt
    save_equilibration: bool = False  # also write the initskip equilibration blocks. Off by default
                                      # so that every block on disk is production: nothing to drop,
                                      # and no need to know the resolved initskip at analysis time.
                                      # They are still integrated and still logged (Rg, energies,
                                      # blow-up checks) -- only the reporter call is skipped.
    max_data_length: int = 100  # blocks the reporter buffers in memory before flushing them to one
                                # blocks_*.h5; larger = fewer, bigger files (and more memory held).
                                # A block is npoly*3 float32 of coordinates, so ~0.8 MB each at
                                # npoly=70000: 100 blocks is ~80 MB buffered per flush.

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
        if self.colrate0 < self.colrate:
            self.colrate0 = self.colrate
        if self.sep is not None and self.sep <= 0:
            raise ValueError("sep must be positive, or None to turn loop extrusion off")
        if self.lifebooststalled <= 0:
            raise ValueError("lifebooststalled must be positive")
        if self.max_data_length < 1:
            raise ValueError("max_data_length must be at least 1 block")
        if self.blocks_per_updater is None:
            if self.extrusion_on:
                # setup() would precalculate one bond list per block and addBond every unique
                # pair it ever sees, for the whole phase -- that is what the chunking is for
                raise ValueError(
                    "blocks_per_updater=None is only allowed with sep=None; with LEFs the bond "
                    "updater precalculates one bond list per block, so give it an explicit value"
                )
        elif self.blocks_per_updater % self.saveevery != 0:
            raise ValueError("saveevery must divide blocks_per_updater ({0})".format(self.blocks_per_updater))
        if (self.monomer_types is None) != (self.interaction_matrix is None):
            raise ValueError(
                "monomer_types and interaction_matrix must be given together "
                "(leave both None for a homopolymer)"
            )
        if self.extrusion_on:
            self.stall_arrays()  # fail here, not several minutes into a run

    # --- derived quantities ---------------------------------------------------------
    @property
    def extrusion_on(self):
        """False when sep is None -- no translocator, no LEF bonds, no bond updater."""
        return self.sep is not None

    @property
    def n_lefs(self):
        if self.sep is None:
            return 0
        return int(self.npoly // self.sep)

    @property
    def poly_steps_per_lef_timestep(self):
        """Polymer steps per LEF timestep; None when extrusion is off.

        A block is poly_steps_per_block polymer steps *and* smc_steps_per_block LEF
        timesteps, so the conversion between the two clocks is just their ratio.
        """
        if not self.extrusion_on:
            return None
        return self.poly_steps_per_block / self.smc_steps_per_block

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

    @staticmethod
    def _schedule_head(sched):
        """'N blocks written ...' phrasing, which depends on save_equilibration."""
        if sched["n_equil_blocks"] == 0:
            return "{0} blocks written".format(sched["n_written"])
        if sched["n_written"] == sched["n_blocks"]:  # equilibration is on disk too
            return "{0} blocks written (the first {1} are equilibration, drop those)".format(
                sched["n_written"], sched["n_equil_blocks"]
            )
        return "{0} blocks written, after {1} equilibration block(s) that are not written".format(
            sched["n_written"], sched["n_equil_blocks"]
        )

    def summary(self):
        """Human-readable rundown of what this configuration will actually do."""
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
        ]
        if self.extrusion_on:
            left, right = self.stall_arrays()
            n_sites = int(((left > 0) | (right > 0)).sum())
            lines += [
                "LEFs         {0} (1 per {1} monomers), lifetime {2:g}, v={3:g}/step".format(
                    self.n_lefs, self.sep, self.life, self.vlef
                ),
                "CTCF         {0} stall site(s), p={1} per encounter, lifetime x{2:g} while stalled".format(
                    n_sites, extrusion.stall_prob_label(left, right), self.lifebooststalled
                ),
                "schedule     {0}, {1} polymer steps and {2} LEF steps each"
                " ({3:g} polymer steps per LEF timestep)".format(
                    self._schedule_head(sched),
                    sched["save_every"] * self.poly_steps_per_block,
                    sched["save_every"] * self.smc_steps_per_block,
                    self.poly_steps_per_lef_timestep,
                ),
            ]
        else:
            lines += [
                "LEFs         none -- loop extrusion is off (sep=None); life/vlef/CTCF are ignored",
                "schedule     {0}, {1} polymer steps each".format(
                    self._schedule_head(sched), sched["save_every"] * self.poly_steps_per_block,
                ),
            ]
        lines.append(
            "chunks       {0} sim rebuild(s) of {1} block(s){2}".format(
                len(sched["chunks"]),
                "/".join(str(n) for n, _ in sched["chunks"][:3]) + ("/..." if len(sched["chunks"]) > 3 else ""),
                "" if self.blocks_per_updater is not None else "  (blocks_per_updater=None)",
            )
        )
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
        elif skip > 0 and self.extrusion_on:
            # make sure equilibration is at least one LEF lifetime long
            while save_every * skip * self.smc_steps_per_block <= self.life:
                skip *= 2

        if self.extrusion_on and self.numsave * save_every * self.smc_steps_per_block <= self.life:
            raise ValueError("the run is shorter than one LEF lifetime; raise numsave or saveevery")

        # the run is split into chunks; each chunk is one Simulation (and one bondUpdater).
        # The equilibration/production boundary always has to fall on a chunk boundary,
        # because colrate switches there and it is baked into the integrator at construction.
        equil_blocks = skip * save_every
        prod_blocks = self.numsave * save_every
        if self.blocks_per_updater is None:
            chunks = [(equil_blocks, False), (prod_blocks, True)]
            chunks = [c for c in chunks if c[0] > 0]
        else:
            bpu = self.blocks_per_updater
            if equil_blocks % bpu != 0:
                raise ValueError("initskip * saveevery must be a multiple of blocks_per_updater")
            if prod_blocks % bpu != 0:
                raise ValueError("numsave * saveevery must be a multiple of blocks_per_updater")
            chunks = [(bpu, False)] * (equil_blocks // bpu) + [(bpu, True)] * (prod_blocks // bpu)

        updater_skip = sum(1 for _, do_save in chunks if not do_save)
        return {
            "save_every": save_every,
            "skip_blocks": skip,
            # (n_blocks_in_chunk, do_save) per Simulation instance
            "chunks": chunks,
            "updater_skip": updater_skip,
            "updater_total": len(chunks),
            # n_blocks counts every block the run produces; n_written counts the ones that
            # reach the trajectory. They differ by the equilibration blocks unless
            # save_equilibration is on, in which case the first n_equil_blocks are the ones to drop.
            "n_blocks": self.numsave + skip,
            "n_equil_blocks": skip,
            "n_written": self.numsave + (skip if self.save_equilibration else 0),
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
    ]
    if params.extrusion_on:
        bits += [
            "life{0:g}".format(params.life),
            "sep{0:g}".format(params.sep),
            "vlef{0:g}".format(params.vlef),
            "dt{0}".format(params.dt),
        ]
        left, right = params.stall_arrays()
        label = extrusion.stall_prob_label(left, right)
        if label is not None:
            bits.append(
                "stallall{0}".format(label) if params.stallall else "stallsites{0}".format(label)
            )
            if params.lifebooststalled != 1.0:
                bits.append("lifeboost{0:g}".format(params.lifebooststalled))
    else:
        bits += ["noLEF", "dt{0}".format(params.dt)]
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
        "repulsionRadius": 1.05,  # this is from old code
        "attractionEnergy": params.attraction_energy,
        "attractionRadius": 2,
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
        save_decimals=2,
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
    if not params.extrusion_on:
        raise ValueError("sep is None, so there is no loop extrusion to build a translocator for")

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
        blocks_*.h5     polychrom trajectory (read with polychrom.hdf5_format.list_URIs);
                        each saved block also carries an "SMCs" (n_lefs, 2) array of LEF
                        leg positions, if save_smc_bonds (extrusion runs only)
        paramsDict.pkl  the SimParams as a dict
        sites.npz       the per-monomer LEF arrays actually used (extrusion runs only)
        bondsAdded.txt  LEF steps taken / new bonds added, if save_smc_bonds

    With ``params.sep is None`` the translocator and the bond updater are skipped
    altogether and this is a plain polymer run on the same block schedule.
    """
    sched = params.schedule()
    folder = make_folder(params, folder)
    if verbose:
        print(params.summary())
        print("\noutput -> {0}".format(folder))

    with open(os.path.join(folder, "paramsDict.pkl"), "wb") as f:
        pickle.dump(params.to_dict(), f)

    bond_updater = None
    if params.extrusion_on:
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
    reporter = HDF5Reporter(folder=folder, max_data_length=params.max_data_length, overwrite=True)

    kbond_wiggle = params.smc_bond_wiggle
    save_every = sched["save_every"]
    save_smc_bonds = params.save_smc_bonds and params.extrusion_on
    prev_step = 0
    tot_bonds_added = 0
    lef_steps_taken = 0
    cur_bonds = []

    try:
        for updater_count, (chunk_blocks, do_save) in enumerate(sched["chunks"]):
            if verbose:
                print("{0} {1} / {2}  ({3} blocks{4})".format(
                    "updater init" if params.extrusion_on else "sim chunk",
                    updater_count, sched["updater_total"], chunk_blocks,
                    "" if do_save else ", equilibration"), flush=True)

            collision_rate = params.colrate if do_save else params.colrate0
            sim = build_simulation(params, polymer, reporter, collision_rate)

            if bond_updater is not None:
                kbond = sim.kbondScalingFactor / (kbond_wiggle ** 2)
                bond_dist = params.smc_bond_dist * sim.length_scale
                bond_updater.setParams({"length": bond_dist, "k": kbond}, {"length": bond_dist, "k": 0})
                cur_bonds, _ = bond_updater.setup(
                    bondForce=sim.force_dict["harmonic_bonds"],
                    blocks=chunk_blocks,
                    smcStepsPerBlock=params.smc_steps_per_block,
                )

            if updater_count in (0, sched["updater_skip"]):
                sim.local_energy_minimization()
            else:
                sim._apply_forces()

            sim.step = prev_step
            for i in range(chunk_blocks):
                if i % save_every == (save_every - 1):
                    # equilibration blocks are still integrated, logged and checked for blow-ups;
                    # save=False only skips handing them to the reporter
                    # cur_bonds is the LEF bond list currently applied to the context, so it goes
                    # into the same h5 block as the conformation it acted on, under "SMCs"
                    extras = {"SMCs": np.array(cur_bonds, dtype=np.int32)} if save_smc_bonds else {}
                    sim.do_block(
                        steps=params.poly_steps_per_block,
                        save=do_save or params.save_equilibration,
                        save_extras=extras,
                    )
                    if save_smc_bonds and (i % (10 * save_every) == (10 * save_every - 1)):
                        with open(os.path.join(folder, "bondsAdded.txt"), "a") as f:
                            f.write("{0} {1}\n".format(lef_steps_taken, tot_bonds_added))
                        tot_bonds_added = 0
                        lef_steps_taken = 0
                else:
                    # skip the GPU->host copy for blocks we are not saving
                    sim.integrator.step(params.poly_steps_per_block)
                    sim.step += params.poly_steps_per_block
                if bond_updater is not None and i < chunk_blocks - 1:
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
