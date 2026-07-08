#!/bin/bash
set -e

echo "======================================================"
echo " 01. PREPROCESSING AND DATASET GENERATION (ETHANOL) "
echo "======================================================"

# The AA trajectory is located in the GROMACS folder
python3 ../../preprocessing/build_cg_dataset.py \
    --traj ../../../GROMACS/ethanol.trr \
    --topol ../../../GROMACS/ethanol.gro \
    --config ethanol_topology.json \
    --output my_ethanol_dataset.bin

echo "Dataset my_ethanol_dataset.bin generato con successo."
