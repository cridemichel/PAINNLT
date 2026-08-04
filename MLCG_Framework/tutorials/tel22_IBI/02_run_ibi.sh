#!/bin/bash
set -e
echo "[2/5] Esecuzione Iterative Boltzmann Inversion (IBI)"

# Esegue il vero loop IBI (Dinamica in ESPResSo + Aggiornamento Henderson)
uv run ../../ibi/run_ibi_loop.py \
    --dataset tel22_dataset.bin \
    --priors cg_priors_seed.json \
    --config test_config.json \
    --rb_info rigid_bodies_info.json \
    --pypresso ../../espresso/build/pypresso \
    --ibi_config ibi_extrapolation_config.json \
    --iterations 3 \
    --outdir ibi_priors
echo "[SUCCESS] IBI completata. Creato ibi_priors/cg_priors_final.json con le tabelle aggiornate."

echo "[INFO] Extrapolation is integrated in run_ibi_loop.py:"
echo "       support-aware update + cosine taper + C1 exponential bond tails."
echo "       Bond tables use a 0.01-5.0 nm safety domain; angles retain conservative endpoint walls."
# uv run python extrapolate_ibi_tables.py
# uv run python cap_ibi_forces.py --max_force 1500.0

echo "[SUCCESS] Tabelle pronte per la sottrazione."
