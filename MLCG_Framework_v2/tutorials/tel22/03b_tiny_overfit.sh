#!/bin/bash
set -euo pipefail

echo "======================================================"
echo " 03b. TINY-SET OVERFIT DIAGNOSTIC"
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
    local torque_weight="$2"
    local model="tiny_${label}.pt"
    local config="tiny_${label}.json"

    rm -f "$model" "$model.manifest.json" cg_training_log.csv

    cat > "$config" <<JSON
{
    "architecture_variant": "painn_canonical_context_silu_v2",
    "num_species": 8,
    "hidden_channels": 64,
    "n_layers": 2,
    "num_rbf": 32,
    "cutoff": 1.2616,

    "toxvaerd_alpha": 0.1,
    "learning_rate": 0.001,
    "epochs": 200,
    "batch_size": 16,
    "torque_weight": ${torque_weight},
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 40,
    "reduce_lr_patience": 15,
    "diagnostic_overfit_frames": 16,
    "include_decoys_in_train": false,
    "shuffle_each_epoch": true
}
JSON

    echo
    echo "------------------------------------------------------"
    echo " Test: ${label} | torque_weight=${torque_weight}"
    echo " 16 frame identici in train e validation"
    echo "------------------------------------------------------"

    "$TRAINER" tel22_dataset.bin "$model" "$config" | tee "tiny_${label}.log"
    mv cg_training_log.csv "tiny_${label}_training_log.csv"
}

run_case "force_only" 0.0
run_case "force_torque" 1.0

echo
echo "Diagnostica completata."
echo "Output:"
echo "  tiny_force_only.log"
echo "  tiny_force_torque.log"
echo "  tiny_force_only_training_log.csv"
echo "  tiny_force_torque_training_log.csv"
