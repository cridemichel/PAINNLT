# Coarse-Grained Dataset Generation Guide

This folder contains the `convert_gro2bin.py` script, a universal tool to convert atomistic GROMACS trajectories (e.g. `.trr`) into a custom binary dataset (`cg_dataset.bin`) used to train the PaiNN coarse-grained neural network potential.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Configuration (`cg_mapping.json`)](#configuration-cg_mappingjson)
4. [Tutorial: Generating a Dataset](#tutorial-generating-a-dataset)
5. [Rigid Body Kinematics](#rigid-body-kinematics)

---

## Overview

The script parses an atomistic MD trajectory and, for each residue defined in a JSON mapping file, computes:
* The **Center of Mass (COM)** of the mapped Coarse-Grained (CG) sites.
* The aggregated forces and torques acting on the rigid body.
* The precise relative geometry of the CG sites, the total mass, and the principal moments of inertia.

It safely unwraps molecules that cross Periodic Boundary Conditions (PBC) to prevent broken coordinates from destroying inertia calculations.

## Prerequisites

You need Python 3 and the `MDAnalysis` library:

```bash
pip install MDAnalysis numpy
```

---

## Configuration (`cg_mapping.json`)

Instead of hardcoding molecules, the script relies on an external configuration file `cg_mapping.json`. This file dictates exactly how all-atom (AA) residues are transformed into CG sites.

A typical configuration file looks like this:

```json
{
    "mapping_method": "COM",
    "residues": {
        "GUA": {  
            "CG_G1": ["N9", "C8"],
            "CG_G2": ["N7", "C5"]
        },
        "ADE": { 
            "CG_ADE": ["*"] 
        }
    },
    "site_types": {
        "CG_G1": 0,
        "CG_G2": 1,
        "CG_ADE": 2
    }
}
```

### JSON Fields Explained:
* `"mapping_method"`: How to collapse the selected atoms. `"COM"` means Center of Mass.
* `"residues"`: A dictionary of GROMACS residue names. 
  * Each key (e.g. `"GUA"`) represents the molecular residue.
  * Its value is a dictionary mapping your custom CG site names (e.g. `"CG_G1"`) to a list of AA atom names (e.g. `["N9", "C8"]`).
  * **Tip:** To map an entire residue to a *single* particle (e.g., Adenine), use `["*"]` as the atom list.
* `"site_types"`: A dictionary assigning a unique integer ID to each custom CG site. This ID is passed to PaiNN as the "atomic number".

---

## Tutorial: Generating a Dataset

Let's assume you ran a simulation of a DNA G-quadruplex (TEL22) in GROMACS and saved the topology as `topologia.tpr` and the trajectory as `traiettoria.trr`.

### Step 1: Create the Mapping
Create a `cg_mapping.json` in the same directory (or use the provided default) mapping out Guanines, Adenines, and Thymines. 

### Step 2: Run the Script
Open your terminal and run:

```bash
python convert_gro2bin.py \
  --topology topologia.tpr \
  --trajectory traiettoria.trr \
  --mapping cg_mapping.json \
  --output cg_dataset.bin
```

*(Note: If you don't provide arguments, the script defaults to the filenames shown above).*

### Step 3: Check the Outputs
The script will output two files:
1. **`cg_dataset.bin`**: The binary trajectory containing mapped coordinates, forces, and torques. Feed this to the C++ training code.
2. **`rigid_bodies_info.json`**: An auto-generated physics file. This file contains the calculated total mass, inertia tensors, and relative site geometry for every residue encountered. You will need this file later when running the ESPResSo MD simulation.

---

## Rigid Body Kinematics

When the script encounters a residue with multiple sites (like `GUA`), it treats it as a Rigid Body. 
* It extracts the masses of all atoms (falling back to standard chemical masses if the `.tpr` parsing fails due to GROMACS version mismatches).
* It computes the exact 3x3 inertia tensor.
* It diagonalizes the tensor to extract the 3 principal moments of inertia (in `amu * nm^2`).
* It computes the distance vector from the rigid body COM to each of its constituent CG sites.

All this data is packed into `rigid_bodies_info.json` so that your MD engine (like ESPResSo) can properly integrate translational and rotational motion using Virtual Sites.
