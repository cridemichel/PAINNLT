#!/bin/bash
set -e
echo "[6/6] Simulazione in ESPResSo con IBI + ML"
../../../espresso/build/pypresso ../../simulation/run_cg_md.py \
    --model tel22_model_ibi.pt \
    --config tel22_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --checkpoint equilibrated.npz \
    --apply_envelope \
    --dt 0.001 \
    --steps 50000
