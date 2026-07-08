#!/bin/bash
set -e
echo "[1/4] Costruzione del Dataset Iniziale"
python ../../preprocessing/build_cg_dataset.py \
    --traj md_whole.trr \
    --topol md.gro \
    --config tel22_topology.json \
    --output tel22_dataset.bin
