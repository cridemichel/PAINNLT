#!/bin/bash
set -e
echo "[1/5] Costruzione del Dataset Iniziale"
uv run ../../preprocessing/build_cg_dataset.py \
    --trajectory md_whole.trr \
    --topology md.gro \
    --config tel22_topology.json \
    --output tel22_dataset.bin
