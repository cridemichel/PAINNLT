#!/bin/bash
set -euo pipefail

# Scan torque_weight after choosing the architecture from 03c_model_ablation.sh.
# Usage example:
#   ./03d_torque_weight_ablation.sh D_both

case_label="${1:-}"
if [ -z "$case_label" ]; then
    echo "Uso: $0 <A_baseline|B_cutoff|C_capacity|D_both>"
    exit 2
fi

base_config="ablation_${case_label}.json"
if [ ! -f "$base_config" ]; then
    echo "Errore: $base_config non trovato. Esegui prima ./03c_model_ablation.sh"
    exit 1
fi
if [ ! -f "tel22_dataset.bin" ]; then
    echo "Errore: tel22_dataset.bin non trovato!"
    exit 1
fi

TRAINER="../../training/build/train_painn"
if [ ! -x "$TRAINER" ]; then
    echo "Errore: trainer non trovato/eseguibile: $TRAINER"
    exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1

run_weight() {
    local tag="$1"
    local weight="$2"
    local config="torque_${case_label}_${tag}.json"
    local model="torque_${case_label}_${tag}.pt"

    python3 - "$base_config" "$config" "$weight" <<'PY'
import json
import sys
src, dst, weight = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(src) as handle:
    cfg = json.load(handle)
cfg["architecture_variant"] = "painn_canonical_context_silu_v2"
cfg["torque_weight"] = weight
cfg["include_decoys_in_train"] = False
cfg["shuffle_each_epoch"] = True
# Keep the comparison controlled and deterministic.
cfg["epochs"] = 100
cfg["batch_size"] = 4
cfg["early_stopping_patience"] = 30
cfg["reduce_lr_patience"] = 10
cfg["diagnostic_overfit_frames"] = 16
with open(dst, "w") as handle:
    json.dump(cfg, handle, indent=4)
    handle.write("\n")
PY

    rm -f "$model" "$model.manifest.json" cg_training_log.csv

    echo
    echo "------------------------------------------------------"
    echo " Architecture: ${case_label} | torque_weight=${weight}"
    echo "------------------------------------------------------"
    "$TRAINER" tel22_dataset.bin "$model" "$config" | tee "torque_${case_label}_${tag}.log"
    mv cg_training_log.csv "torque_${case_label}_${tag}_training_log.csv"
}

run_weight "w025" 0.25
run_weight "w050" 0.50
run_weight "w100" 1.00

python3 ./summarize_training_grid.py \
    --prefix "torque_${case_label}" \
    --output "torque_${case_label}_summary.csv" \
    --epoch-metric balanced_ft

echo
echo "Torque-weight scan completato."
echo "Scegli il compromesso F/T, non necessariamente il minimo totale: torque_weight cambia per definizione la scala della loss totale."
