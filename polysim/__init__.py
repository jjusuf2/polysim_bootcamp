"""Polymer + loop-extrusion simulations, built on polychrom / OpenMM.

Install once, in editable mode, from the repository root::

    pip install -e .

Then every notebook can import from anywhere, regardless of its working directory::

    from polysim import extrusion, OUTPUTS
    from polysim.sim3d import SimParams, run

Because the install is editable, edits to these files take effect immediately -- there
is nothing to reinstall after a ``git pull``. Inside a running Jupyter kernel, put

    %load_ext autoreload
    %autoreload 2

at the top of the notebook to pick up edits without restarting. The one exception is
``LEF_Dynamics.pyx``: pyximport recompiles it when it changes, but a compiled extension
module cannot be swapped inside a live kernel, so restart the kernel after editing it.
"""

import os
from pathlib import Path

__version__ = "0.1.0"

#: Where simulation output goes. Deliberately outside the repository so results never
#: land in git. Override per run with the ``outpath`` parameter, or by setting the
#: POLYSIM_OUTPUTS environment variable.
OUTPUTS = Path(os.environ.get("POLYSIM_OUTPUTS", "/mnt/md1/jjusuf/polysim/outputs"))

#: The repository root, useful for locating data files that live alongside the code.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Submodules are deliberately not imported here: `from polysim import extrusion` pulls
# it in on demand, so importing OUTPUTS alone does not compile the Cython extension, and
# the 1D notebook never has to import polychrom/OpenMM (which only sim3d needs).

__all__ = ["OUTPUTS", "REPO_ROOT", "__version__"]
