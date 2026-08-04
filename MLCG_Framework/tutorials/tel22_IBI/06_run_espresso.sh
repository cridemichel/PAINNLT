#!/bin/bash
set -e
echo "[6/6] Simulazione in ESPResSo con IBI + ML"
PYTHONUNBUFFERED=1 ../../espresso/build/pypresso ../../simulation/run_cg_md.py \
    --model tel22_model_ibi_v2.pt \
    --config tel22_training_config.json \
    --priors ibi_priors/cg_priors_final.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --checkpoint equilibrated.npz \
    --dt 0.0001 \
    --steps 100000 \
    --nve \
    --log_interval 100
