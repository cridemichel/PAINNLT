#!/bin/bash
set -e
echo "[5/6] Equilibrazione del sistema (Steepest Descent + Warmup)"
../../espresso/build/pypresso ../../simulation/equilibrate.py \
    --priors_only \
    --config tel22_training_config.json \
    --priors ibi_priors/cg_priors_final.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --out_checkpoint equilibrated.npz
echo "[SUCCESS] Sistema equilibrato e salvato in equilibrated.npz"
