# Polymer simulation bootcamp code

Welcome to James' **Hansen Lab Polymer Simulation Bootcamp**! This repository contains code to run and analyze 3D molecular dynamics (MD) simulations of chromatin polymers with loop extrusion and sticky interactions, coarse-grained at the kilobase scale. These simulations can be used to model chromatin dynamics, compartmentalization, CTCF & cohesin, enhancer-promoter interactions, and more. These models use the [polychrom](https://github.com/open2c/polychrom) package (Mirny Lab/Open2C), which is a wrapper on the industry-standard MD simulation package [OpenMM](https://openmm.org/).

This repository also serves as a consolidated resource for polymer simulation code as it is used by the Hansen Lab. Here we introduce the `polysim` package, which re-implements the key elements of many standalone simulation scripts from lab projects from 2021-2026 with an emphasis on user-friendliness and code readability. The `polysim` package therefore serves as a great starting point for your next polymer simulation project. For more details on how our code relates to existing scripts and packages, see the Provenance section below.

## Setup

To get started, make a conda environment with the following packages (with recommended version numbers):
* Essential for running simulations
    * python 3.12.2
    * numpy 1.26.4
    * openmm 8.1.1
    * polychrom 0.1.1
    * cudatoolkit 11.8.0
* Other packages for analyses in this tutorial
    * matplotlib 3.11.1
    * tqdm 4.70.0
    * ipykernel 7.2.0
    * py3Dmol 2.5.5
    * noctiluca 0.1.4

These versions are appropriate for running the simulation code on the Hansen Lab computers' GPU's, which as of August 2026 are:
* rosalind: NVIDIA RTX 4500, CUDA 12.9
* florence: NVIDIA RTX 2080, CUDA 13.0
* katherine/joan: NVIDIA RTX 6000, CUDA 13.2

**Quick-start:** If you are using a lab computer, just run the commands below.

**1.** Clone this repository

```
$ cd /mnt/md0/<username>/...  # choose an appropriate location

$ git clone https://github.com/jjusuf2/polysim_bootcamp
```

**2.** Create an environment with the necessary packages (conda packages are listed in `polysim_environment_explicit.txt`)


```bash
$ conda create -n polysim -c conda-forge --file polysim_bootcamp/polysim_environment_explicit.txt

$ conda activate polysim

$ pip install py3Dmol==2.5.5 noctiluca==0.1.4 "polychrom @ git+https://github.com/open2c/polychrom.git@4c9e3f8"
```

**3.** Install the `polysim` package 
```
$ pip install -e polysim_bootcamp
```

## Tutorial

You will find all the code you need to follow the bootcamp in the `tutorial` directory.

To run a simulation:
* `sim3D_complete.py`
* `sim3D_naked_chain_for_calibration.py`

To analyze the results:
* `basic_analysis.ipynb`
* `microc.ipynb`
* `calibration.ipynb`
* `visualization.ipynb`


## Provenance

Lots of different simulation code has been written over the years, and each version contains unique features for the specific biological process of interest. Here, we provide the `polysim` package: a revised, simplified codebase to run 3D polymer simulations on `polychrom` with two key features:
* sticky sites, to model _cis_-regulatory elements or compartments
* CTCF/cohesin-mediated loop extrusion using a coupled 1D simulation.

Below we describe how each script in the package relates to existing code:

* Back-end

  * `LEF_dynamics.pyx` implements the 1D simulations of cohesin motion. Is almost identical to the script of the same name in `polychrom`'s [loop extrusion example](https://github.com/open2c/polychrom/tree/master/examples/loopExtrusion), but with some bug fixes (see comments in script for more details). A [similar script](https://github.com/mirnylab/microcompartments/tree/main/m-to-g1) was used in the simulation code from Goel et al. 2026, and a [distantly related version](https://github.com/ahansenlab/DNA_break_synapsis_models/blob/master/DSB_smcTranslocator_v2.pyx) was written to model DSBs in Yang et al. 2023 and was used verbatim in Jusuf et al. 2026 and Ramanathan et al. 2026.

  * `bond_updater.py` defines the `smcBondUpdater` class which takes the results of the 1D simulation (LEF positions) and implements them in the 3D simulation. It is a simplified version of [this script](https://github.com/mirnylab/microcompartments/blob/main/m-to-g1/smcBondUpdater.py) from Goel et al. 2026, which in turn was adapted from a `polychrom` example.

* Front-end (importable modules)

  * `extrusion.py` contains key functions to set up and run a stand-alone 1D simulation of cohesin motion.

  * `sim3D.py` contains key functions to set up and run the full 3D simulation, which can include a coupled 1D simulation.

  These two scripts essentially modularize the [main simulation script](https://github.com/mirnylab/microcompartments/blob/main/m-to-g1/m-to-g1_transition.py) from Goel et al. 2026 into functions that can be adapted to model different loci.

We especially acknowledge E. J. Banigan, J. H. Yang, H. B. Brandão, and M. Imakaev for their substantial contributions to previous simulation code that has evolved into the version presented here.