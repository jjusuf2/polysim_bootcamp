# Polymer simulation bootcamp

`main_example` contains code to perform 1D and 3D polymer simulations. It is primarily based on the polychrom example, with a few recent bug fixes, but with the organizational structure of Ed's code.

`LEF_Dynamics.pyx` is directly from polychrom, with bug fixes
`simUtils.py`, `smcBondUpdater.py`, and `tools.py` contain helper functions from Ed's microcompartment code

## 3D simulations

Open `main_example/run_sim3D.ipynb`. It walks through polymer geometry, CTCF sites,
optional monomer types, and the run itself. The code behind it:

| file | role |
| --- | --- |
| `extrusion.py` | CTCF stall arrays and the 1D LEF translocator; no OpenMM needed |
| `sim3D.py` | importable module: `SimParams`, polymer/force setup, run loop |
| `smcBondUpdater.py` | pushes LEF positions into OpenMM harmonic bonds |

`sim3D.py` no longer models compartments and no longer parses a command line or a