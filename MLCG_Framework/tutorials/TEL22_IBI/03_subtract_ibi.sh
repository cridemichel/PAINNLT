#!/bin/bash
set -e

echo "========================================="
echo "Generating Final Residual Dataset (IBI)"
echo "========================================="

# This script runs build_cg_dataset.py again, but this time passing the 
# IBI-updated cg_priors.json so that the tabulated forces are subtracted!
uv run ../../preprocessing/build_cg_dataset.py \
    --topology md.gro \
    --trajectory md_whole.trr \
    --config tel22_topology.json \
    --priors cg_priors.json \
    --output tel22_dataset_ibi.bin

echo "[SUCCESS] Final residual dataset created as tel22_dataset_ibi.bin"
