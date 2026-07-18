#!/bin/bash
set -e

echo "======================================================"
echo " 02. PREPROCESSING AND DATASET GENERATION "
echo "======================================================"

if [ ! -f "md.trr" ] || [ ! -f "md.gro" ]; then
    echo "Errore: md.trr o md.gro non trovati! Hai eseguito 01_run_gromacs.sh?"
    exit 1
fi

/Users/demichel/PYTHON/bin/python ../../preprocessing/build_cg_dataset.py \
    --topology md.gro \
    --trajectory md_whole.trr \
    --config tel22_topology.json \
    --output tel22_dataset.bin

echo "Dataset tel22_dataset.bin generato con successo."
