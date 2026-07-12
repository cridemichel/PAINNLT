#!/bin/bash
set -e
echo "[2/5] Esecuzione Iterative Boltzmann Inversion (IBI)"

# Esegue il vero loop IBI (Dinamica in ESPResSo + Aggiornamento Henderson)
uv run ../../ibi/run_ibi_loop.py \
    --dataset tel22_dataset.bin \
    --priors cg_priors.json \
    --iterations 3 \
    --outdir ibi_priors

echo "[SUCCESS] IBI completata. cg_priors.json e le tabelle sono state aggiornate."
