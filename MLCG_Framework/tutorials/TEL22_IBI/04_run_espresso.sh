#!/bin/bash
set -e
echo "[4/4] Simulazione in ESPResSo con IBI + ML"
../../../espresso/build/pypresso ../../simulation/run_cg_md.py \
    --model tel22_model_ibi.pt \
    --config tel22_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --dt 0.002 \
    --steps 1000
