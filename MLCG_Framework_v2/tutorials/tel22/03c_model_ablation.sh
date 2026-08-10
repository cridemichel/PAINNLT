#!/bin/bash
set -euo pipefail

# Controlled tiny-set ablation motivated by the TEL22 diagnostics:
# A: current model
# B: cutoff only
# C: capacity/RBF only
# D: cutoff + capacity/RBF
#
# All cases use the same 16 deterministic frames for train and validation.
# This is a representability/optimization test, NOT a generalization estimate.

echo "======================================================"
echo " 03c. TEL22 MODEL ABLATION (TINY SET)"
echo "======================================================"

if [ ! -f "tel22_dataset.bin" ]; then
    echo "Errore: tel22_dataset.bin non trovato!"
    exit 1
fi

TRAINER="../../training/build/train_painn"
if [ ! -x "$TRAINER" ]; then
    echo "Errore: trainer non trovato/eseguibile: $TRAINER"
    echo "Ricompila prima con: cd ../../training/build && cmake .. && make -j"
    exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1

run_case() {
    local label="$1"
    local hidden="$2"
    local layers="$3"
    local rbf="$4"
    local cutoff="$5"
    local model="ablation_${label}.pt"
    local config="ablation_${label}.json"

    rm -f "$model" "$model.manifest.json" cg_training_log.csv

    cat > "$config" <<JSON
{
    "num_species": 8,
    "hidden_channels": ${hidden},
    "n_layers": ${layers},
    "num_rbf": ${rbf},
    "cutoff": ${cutoff},

    "toxvaerd_alpha": 0.1,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 4,
    "torque_weight": 1.0,
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 30,
    "reduce_lr_patience": 10,
    "diagnostic_overfit_frames": 16
}
JSON

    echo
    echo "------------------------------------------------------"
    echo " Case ${label}: hidden=${hidden}, layers=${layers}, rbf=${rbf}, cutoff=${cutoff} nm"
    echo " torque_weight=1.0 | 16 identical train/validation frames"
    echo "------------------------------------------------------"

    "$TRAINER" tel22_dataset.bin "$model" "$config" | tee "ablation_${label}.log"
    mv cg_training_log.csv "ablation_${label}_training_log.csv"
}

run_case "A_baseline" 64 2 32 1.2616
run_case "B_cutoff"   64 2 32 1.6000
run_case "C_capacity" 128 3 64 1.2616
run_case "D_both"     128 3 64 1.6000

python3 ./summarize_training_grid.py \
    --prefix ablation \
    --output model_ablation_summary.csv

echo
echo "Ablation completata."
echo "Confronta soprattutto val_F e val_T in model_ablation_summary.csv."
echo "Non promuovere automaticamente il best case a produzione: il tiny set misura representability, non generalizzazione."
