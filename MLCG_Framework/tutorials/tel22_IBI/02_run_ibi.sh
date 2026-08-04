#!/bin/bash
set -e
echo "[2/5] Esecuzione Iterative Boltzmann Inversion (IBI)"

# Esegue il vero loop IBI (Dinamica in ESPResSo + Aggiornamento Henderson)
uv run ../../ibi/run_ibi_loop.py \
    --dataset tel22_dataset.bin \
    --priors cg_priors.json \
    --config test_config.json \
    --rb_info rigid_bodies_info.json \
    --iterations 3 \
    --outdir ibi_priors

echo "[SUCCESS] IBI completata. cg_priors.json e le tabelle sono state aggiornate."

echo "[INFO] Capping and extrapolation is now handled robustly inside run_ibi_loop.py using CubicSplines!"
# uv run python extrapolate_ibi_tables.py
# uv run python cap_ibi_forces.py --max_force 1500.0

echo "[SUCCESS] Tabelle pronte per la sottrazione."
