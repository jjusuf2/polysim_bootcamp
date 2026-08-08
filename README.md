# Polymer simulation bootcamp code

Welcome to James' **Hansen Lab Polymer Simulation Bootcamp**! This repository contains code to run and analyze 3D molecular dynamics (MD) simulations of chromatin polymers with loop extrusion and sticky interactions, coarse-grained at the kilobase scale. These simulations can be used to model chromatin dynamics, compartmentalization, CTCF & cohesin, enhancer-promoter interactions, and more. These models use the [polychrom](https://github.com/open2c/polychrom) package (Mirny Lab/Open2C), which is a wrapper on the industry-standard MD simulation package [OpenMM](https://openmm.org/).

This repository also serves as a consolidated resource for polymer simulation code as it is used by the Hansen Lab. Here we introduce the `polysim` package, which re-implements the key elements of many standalone simulation scripts from lab projects from 2021-2026 with an emphasis on user-friendliness and code readability. The `polysim` package therefore serves as a great starting point for your next polymer simulation project. For more details on how our code relates to existing scripts and packages, see the Provenance section below.

## Getting started

You will find all the code you need to follow the bootcamp in the `tutorial` directory. Please make a conda environment with the following packages (with recommended version numbers):
* Essential for simulations
    * python 3.12.2
    * numpy 1.26.4
    * openmm 8.1.1
    * polychrom 0.1.1
    * cudatoolkit 11.8.0
* Other packages used in tutorial
    * matplotlib 3.11.1
    * tqdm 4.70.0
    * ipykernel 7.2.0
    * py3Dmol 2.5.5

If you are using a lab computer, you can do this by cloning James' conda environment exactly using the commands below. This will be much faster than creating an environment from scratch, but it might not work on other operating systems.
```bash
$ conda create -n polysim -c conda-forge --file /mnt/md0/jjusuf/environments/polysim3_exact_2026aug.txt

$ conda run -n polysim pip install tqdm==4.70.0 matplotlib==3.11.1 py3Dmol==2.5.5 "polychrom @ git+https://github.com/open2c/polychrom.git@4c9e3f8"

$ conda activate polysim
```

The code here is designed to be run on the Hansen Lab computers' GPU's, which as of August 2026 are:
* rosalind: NVIDIA RTX 4500, CUDA 12.9
* florence: NVIDIA RTX 2080, CUDA 13.0
* katherine/joan: NVIDIA RTX 6000, CUDA 13.2

## 3D simulations

Open `main_example/run_sim3D.ipynb`. It walks through polymer geometry, CTCF sites,
optional monomer types, and the run itself. The code behind it:

| file | role |
| --- | --- |
| `extrusion.py` | CTCF stall arrays and the 1D LEF translocator; no OpenMM needed |
| `sim3D.py` | importable module: `SimParams`, polymer/force setup, run loop |
| `smcBondUpdater.py` | pushes LEF positions into OpenMM harmonic bonds |

`sim3D.py` no longer models compartments and no longer parses a command line or a

## Provenance

`main_example` contains code to perform 1D and 3D polymer simulations. It is primarily based on the polychrom example, with a few recent bug fixes, but with the organizational structure of Ed's code.

`LEF_Dynamics.pyx` is directly from polychrom, with bug fixes
`simUtils.py`, `smcBondUpdater.py`, and `tools.py` contain helper functions from Ed's microcompartment code
