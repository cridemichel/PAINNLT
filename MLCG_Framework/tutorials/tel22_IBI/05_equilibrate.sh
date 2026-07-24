#!/bin/bash
set -e
echo "[5/6] Equilibrazione del sistema (Steepest Descent + Warmup)"
../../espresso/build/pypresso ../../simulation/equilibrate.py \
    --model tel22_model_ibi.pt \
    --config tel22_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --out_checkpoint equilibrated.npz
echo "[SUCCESS] Sistema equilibrato e salvato in equilibrated.npz"
