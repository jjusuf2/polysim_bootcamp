"""Loop-extrusion setup: CTCF stall arrays and the 1D LEF translocator.

Nothing in here needs OpenMM or polychrom, so it can be imported on its own (e.g. to
inspect a CTCF layout before committing to a 3D run).

The translocator itself lives in ``LEF_Dynamics.pyx`` and is built on import via
pyximport. Its constructor takes five per-monomer arrays plus the number of LEFs:

    birthArray       relative loading probability per monomer
    deathArray       p(unbind per timestep) for a LEF leg that is *not* stalled
    stallLeftArray   p(stall per arrival) for the left (leftward-moving) leg
    stallRightArray  p(stall per arrival) for the right (rightward-moving) leg
    pauseArray       p(do not step per timestep); p(step) = 1 - pause
    stallDeathArray  p(unbind per timestep) for a leg that *is* stalled

``death()`` in the translocator takes ``max(falloff_left, falloff_right)``, so a LEF's
lifetime is only actually extended once *both* legs are stalled.
"""

import os
import sys

import numpy as np

# LEF_Dynamics.pyx sits next to this file; make it importable even when the notebook
# or script was started from some other working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from LEF_Dynamics import LEFTranslocatorDirectional


def sites_to_array(length, sites, prob):
    """Per-monomer probability array from a list of indices, or a {index: prob} dict.

    ``sites`` may be None (all zeros), a sequence of monomer indices (each gets ``prob``),
    or a dict mapping monomer index -> its own probability. ``prob`` is only read for the
    sequence form, so it may be None whenever every site carries its own probability.
    """
    arr = np.zeros(length, dtype=np.double)
    if sites is None:
        return arr

    if isinstance(sites, dict):
        pairs = [(int(k), float(v)) for k, v in sites.items()]
    else:
        if prob is None:
            raise ValueError(
                "stall probability is None but the CTCF sites were given as a plain list, "
                "which has no per-site probabilities; either set stall to a number or pass "
                "the sites as a {monomer index: probability} dict"
            )
        pairs = [(int(s), float(prob)) for s in np.asarray(sites, dtype=int).ravel()]

    for site, p in pairs:
        if not 0 <= site < length:
            raise ValueError("CTCF site {0} is outside the chain [0, {1})".format(site, length))
        if not 0.0 <= p <= 1.0:
            raise ValueError("stall probability {0} at site {1} is not in [0, 1]".format(p, site))
        arr[site] = p
    return arr


def build_stall_arrays(length, ctcf_left=None, ctcf_right=None, stall_prob=None, stall_all=False):
    """CTCF stall probabilities for the two LEF legs.

    Parameters
    ----------
    length : int
        Number of monomers.
    ctcf_left, ctcf_right : sequence of int, or dict {index: prob}, or None
        ``stallLeft`` is read at the left (leftward-moving) leg's position and
        ``stallRight`` at the right (rightward-moving) leg's. So to hold a loop between
        monomers a < b, put ``a`` in ``ctcf_left`` and ``b`` in ``ctcf_right``.
    stall_prob : float or None
        Probability applied at each site given as a plain index. It is ignored for dict
        input, so pass None when both site lists are dicts of per-site probabilities.
    stall_all : bool
        Stall everywhere at ``stall_prob``; overrides the site lists, and so needs
        ``stall_prob`` to be a number.
    """
    if stall_all:
        if stall_prob is None:
            raise ValueError("stall_all=True stalls every monomer at stall_prob, which cannot be None")
        left = np.full(length, float(stall_prob), dtype=np.double)
        right = np.full(length, float(stall_prob), dtype=np.double)
        return left, right

    left = sites_to_array(length, ctcf_left, stall_prob)
    right = sites_to_array(length, ctcf_right, stall_prob)
    return left, right


def stall_prob_label(stall_left, stall_right):
    """Short description of the stall probabilities in use, for summaries and folder names.

    ``'0.8'`` when every stall site shares one probability, ``'0.05-1'`` when they differ,
    and None when there are no stall sites at all.
    """
    values = np.concatenate([stall_left[stall_left > 0.0], stall_right[stall_right > 0.0]])
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return "{0:g}".format(lo)
    return "{0:g}-{1:g}".format(lo, hi)


def build_lef_arrays(
    length,
    lifetime,
    vlef,
    stall_left,
    stall_right,
    life_boost_stalled=1.0,
    birth_prob=0.1,
):
    """The five per-monomer arrays the translocator needs, as a dict.

    ``life_boost_stalled`` multiplies a LEF's lifetime while it is stalled: the stalled
    falloff rate is set to ``1 / (lifetime * life_boost_stalled)`` at the CTCF sites. A
    leg only ever sits stalled where its stall probability is nonzero, so lowering the
    rate at those sites is exactly "stalled LEFs live longer". Use 1.0 for no boost.
    """
    if lifetime <= 0:
        raise ValueError("lifetime must be positive")
    if not 0.0 < vlef <= 1.0:
        raise ValueError("vlef is a per-timestep step probability and must be in (0, 1]")
    if life_boost_stalled <= 0:
        raise ValueError("life_boost_stalled must be positive")

    birth = np.zeros(length, dtype=np.double) + birth_prob
    death = np.zeros(length, dtype=np.double) + 1.0 / lifetime
    stall_death = np.zeros(length, dtype=np.double) + 1.0 / lifetime
    # the translocator tests `randnum() > pause` to step, so p(step per leg per timestep) = vlef
    pause = np.ones(length, dtype=np.double) * (1.0 - vlef)

    stall_sites = (stall_left > 0.0) | (stall_right > 0.0)
    stall_death[stall_sites] = 1.0 / (lifetime * life_boost_stalled)

    return {
        "birth": birth,
        "death": death,
        "stallLeft": stall_left,
        "stallRight": stall_right,
        "pause": pause,
        "stallDeath": stall_death,
    }


def make_translocator(arrays, n_lefs):
    """Build the LEFTranslocatorDirectional from the dict returned by build_lef_arrays."""
    if n_lefs < 1:
        raise ValueError("need at least one LEF; check npoly / sep")
    return LEFTranslocatorDirectional(
        arrays["birth"],
        arrays["death"],
        arrays["stallLeft"],
        arrays["stallRight"],
        arrays["pause"],
        arrays["stallDeath"],
        n_lefs,
    )
