#!/bin/bash
set -euo pipefail

echo "[5/6] Equilibrazione priors-only e IBI+ML"

PYPRESSO="../../espresso/build/pypresso"
EQUILIBRATE="../../simulation/equilibrate.py"
COMMON_ARGS=(
    --config tel22_training_config.json
    --priors ibi_priors/cg_priors_final.json
    --rb_info rigid_bodies_info.json
    --dataset tel22_dataset.bin
    --dt 0.0001
)

echo "[INFO] Generating priors-only checkpoint for the control scaling test..."
"${PYPRESSO}" "${EQUILIBRATE}"     --priors_only     "${COMMON_ARGS[@]}"     --out_checkpoint equilibrated_priors.npz

echo "[INFO] Equilibrating under the final IBI + PaiNN Hamiltonian..."
"${PYPRESSO}" "${EQUILIBRATE}"     --model tel22_model_ibi_v2.pt     --checkpoint equilibrated_priors.npz     "${COMMON_ARGS[@]}"     --steps_sd 0     --steps_md 0     --out_checkpoint equilibrated_ml.npz

echo "[SUCCESS] Checkpoints written: equilibrated_priors.npz, equilibrated_ml.npz"
