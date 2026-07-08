#!/bin/bash
set -e

echo "======================================================"
echo " 01. PREPROCESSING AND DATASET GENERATION (WATER) "
echo "======================================================"

python3 ../../preprocessing/build_cg_dataset.py \
    --traj traiettoria.trr \
    --topol conf.gro \
    --config water_topology.json \
    --output my_water_dataset.bin

echo "Dataset my_water_dataset.bin generato con successo."
