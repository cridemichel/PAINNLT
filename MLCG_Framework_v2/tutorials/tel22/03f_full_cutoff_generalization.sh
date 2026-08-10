#!/bin/bash
set -euo pipefail

# Full-dataset generalization test for TEL22.
# Unlike 03c/03d/03e, this is NOT a tiny-set overfit test:
#   - physical frames are split deterministically into train/validation
#   - zero-target OOD decoys are kept in train only
#   - both cutoff cases use the same split seed and same model seed
#
# Default: 10 epochs per case. Override, e.g.:
#   FULL_CUTOFF_TEST_EPOCHS=15 ./03f_full_cutoff_generalization.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER="$ROOT_DIR/training/build/train_painn"
DATASET="$SCRIPT_DIR/tel22_dataset.bin"
BASE_CONFIG="$SCRIPT_DIR/tel22_training_config.json"
RUN_ROOT="$SCRIPT_DIR/full_cutoff_generalization_runs"
EPOCHS="${FULL_CUTOFF_TEST_EPOCHS:-10}"

if [ ! -x "$TRAINER" ]; then
    echo "Errore: trainer non trovato/eseguibile: $TRAINER"
    echo "Questa patch modifica training/train_painn.cpp: ricompila prima con"
    echo "  cd $ROOT_DIR/training/build && cmake .. && make -j"
    exit 1
fi
if [ ! -f "$DATASET" ]; then
    echo "Errore: dataset non trovato: $DATASET"
    exit 1
fi
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Errore: config base non trovata: $BASE_CONFIG"
    exit 1
fi
case "$EPOCHS" in
    ''|*[!0-9]*) echo "Errore: FULL_CUTOFF_TEST_EPOCHS deve essere un intero positivo"; exit 2 ;;
esac
if [ "$EPOCHS" -le 0 ]; then
    echo "Errore: FULL_CUTOFF_TEST_EPOCHS deve essere > 0"
    exit 2
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$RUN_ROOT"

echo "======================================================"
echo " 03f. FULL-DATASET CUTOFF GENERALIZATION TEST"
echo "======================================================"
echo "Architecture: hidden=128 | layers=3 | rbf=64 | torque_weight=0.5"
echo "Cases: cutoff 1.2616 vs 1.6000 nm"
echo "Epochs per case: $EPOCHS"
echo "Validation: PHYSICAL ONLY | legacy decoys: EXCLUDED | split_seed=42"
echo

run_case() {
    local label="$1"
    local cutoff="$2"
    local case_dir="$RUN_ROOT/$label"
    local config="$case_dir/config.json"

    rm -rf "$case_dir"
    mkdir -p "$case_dir"

    python3 - "$BASE_CONFIG" "$config" "$cutoff" "$EPOCHS" <<'PY'
import json
import sys
src, dst, cutoff, epochs = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
with open(src) as handle:
    cfg = json.load(handle)
# Fix the architecture/force-torque objective; vary ONLY cutoff.
cfg.update({
    "num_species": 8,
    "hidden_channels": 128,
    "n_layers": 3,
    "num_rbf": 64,
    "cutoff": cutoff,
    "learning_rate": 0.001,
    "epochs": epochs,
    "batch_size": 4,
    "torque_weight": 0.5,
    "grad_clip_norm": 1.0,
    "weight_decay": 0.0,
    "lipschitz_lambda": 0.0,
    "diagnostic_overfit_frames": 0,
    "physical_validation_only": True,
    "include_decoys_in_train": False,
    "shuffle_each_epoch": True,
    "split_seed": 42,
    "validation_fraction": 0.2,
    # Do not let LR scheduling / early stopping confound this short A/B test.
    "early_stopping_patience": max(50, epochs + 5),
    "reduce_lr_patience": max(50, epochs + 5),
})
with open(dst, "w") as handle:
    json.dump(cfg, handle, indent=4)
    handle.write("\n")
PY

    echo "------------------------------------------------------"
    echo " Case $label | cutoff=$cutoff nm | epochs=$EPOCHS"
    echo "------------------------------------------------------"
    (
        cd "$case_dir"
        "$TRAINER" "$DATASET" model.pt config.json | tee run.log
    )

    # TEL22 guardrail for the corrected dataset: 1001 physical frames, no legacy decoys.
    if ! grep -q "Detected physical frames: 1001" "$case_dir/run.log"; then
        echo "[ERROR] $label: attesi 1001 frame fisici. Controlla il riconoscimento decoy."
        exit 3
    fi
    if ! grep -q "Detected zero-target OOD decoys: 0" "$case_dir/run.log"; then
        echo "[ERROR] $label: attesi 0 legacy decoy. Controlla il riconoscimento decoy."
        exit 3
    fi
}

run_case "A_cutoff_1p2616" 1.2616
run_case "B_cutoff_1p6000" 1.6000

python3 "$SCRIPT_DIR/summarize_full_cutoff_generalization.py" \
    --run-root "$RUN_ROOT" \
    --output "$SCRIPT_DIR/full_cutoff_generalization_summary.csv"

echo
echo "Test completato."
echo "Risultato: $SCRIPT_DIR/full_cutoff_generalization_summary.csv"
echo "Interpretazione: improvement_vs_zero > 0 significa migliore del predittore zero sulla validation fisica."
