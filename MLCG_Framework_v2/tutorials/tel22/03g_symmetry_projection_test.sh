#!/bin/bash
set -euo pipefail

# Full-dataset raw-vs-symmetry-projected target diagnostic.
# Uses the fast PaiNN baseline (64x2, 32 RBF, cutoff 1.2616 nm) and the same
# physical-only validation split in both cases. Legacy unmasked OOD decoys are
# excluded from optimization. The original tel22_dataset.bin is NEVER modified.
#
# Override runtime if desired:
#   SYMPROJ_TEST_EPOCHS=15 SYMPROJ_TEST_BATCH=16 ./03g_symmetry_projection_test.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINER="$ROOT_DIR/training/build/train_painn"
RAW_DATASET="$SCRIPT_DIR/tel22_dataset.bin"
BASE_CONFIG="$SCRIPT_DIR/tel22_training_config.json"
RUN_ROOT="$SCRIPT_DIR/symmetry_projection_runs"
PROJECTED_DATASET="$RUN_ROOT/tel22_dataset_symmetry_projected.bin"
PROJECTION_REPORT="$RUN_ROOT/symmetry_projection_report.json"
EPOCHS="${SYMPROJ_TEST_EPOCHS:-10}"
BATCH="${SYMPROJ_TEST_BATCH:-16}"

for value_name in EPOCHS BATCH; do
    value="${!value_name}"
    case "$value" in
        ''|*[!0-9]*) echo "Errore: $value_name deve essere un intero positivo"; exit 2 ;;
    esac
    if [ "$value" -le 0 ]; then
        echo "Errore: $value_name deve essere > 0"
        exit 2
    fi
done

if [ ! -x "$TRAINER" ]; then
    echo "Errore: trainer non trovato/eseguibile: $TRAINER"
    exit 1
fi
if [ ! -f "$RAW_DATASET" ]; then
    echo "Errore: dataset non trovato: $RAW_DATASET"
    exit 1
fi
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Errore: config base non trovata: $BASE_CONFIG"
    exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

printf '%s\n' \
  "======================================================" \
  " 03g. RAW vs SYMMETRY-PROJECTED TARGET TEST" \
  "======================================================" \
  "Fast architecture: hidden=64 | layers=2 | rbf=32 | cutoff=1.2616 nm" \
  "torque_weight=0.5 | batch=$BATCH | epochs/case=$EPOCHS" \
  "Validation: PHYSICAL ONLY | legacy decoys: EXCLUDED | split_seed=42" \
  "Original dataset is read-only; projected targets go to $RUN_ROOT" \
  ""

python3 "$SCRIPT_DIR/project_symmetry_targets.py" \
    "$RAW_DATASET" "$PROJECTED_DATASET" --report "$PROJECTION_REPORT"

python3 - "$PROJECTION_REPORT" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    r = json.load(fh)
c = r["counts"]
if c["physical_frames"] != 1001 or c["zero_target_decoys_unchanged"] != 0:
    raise SystemExit(
        f"[ERROR] TEL22 guardrail failed: physical={c['physical_frames']} decoy={c['zero_target_decoys_unchanged']} (expected 1001/0)"
    )
PY

run_case() {
    local label="$1"
    local dataset="$2"
    local case_dir="$RUN_ROOT/$label"
    local config="$case_dir/config.json"
    mkdir -p "$case_dir"

    python3 - "$BASE_CONFIG" "$config" "$EPOCHS" "$BATCH" <<'PY'
import json, sys
src, dst, epochs, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src) as fh:
    cfg = json.load(fh)
cfg.update({
    "num_species": 8,
    "hidden_channels": 64,
    "n_layers": 2,
    "num_rbf": 32,
    "cutoff": 1.2616,
    "learning_rate": 0.001,
    "epochs": epochs,
    "batch_size": batch,
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
    "early_stopping_patience": max(50, epochs + 5),
    "reduce_lr_patience": max(50, epochs + 5),
})
with open(dst, "w") as fh:
    json.dump(cfg, fh, indent=4)
    fh.write("\n")
PY

    echo "------------------------------------------------------"
    echo " Case $label | dataset=$(basename "$dataset")"
    echo "------------------------------------------------------"
    (
        cd "$case_dir"
        "$TRAINER" "$dataset" model.pt config.json | tee run.log
    )

    grep -q "Detected physical frames: 1001" "$case_dir/run.log" || {
        echo "[ERROR] $label: attesi 1001 frame fisici"; exit 3; }
    grep -q "Detected zero-target OOD decoys: 0" "$case_dir/run.log" || {
        echo "[ERROR] $label: attesi 0 legacy decoy"; exit 3; }
}

run_case "A_raw" "$RAW_DATASET"
run_case "B_projected" "$PROJECTED_DATASET"

python3 "$SCRIPT_DIR/summarize_symmetry_projection.py" \
    --run-root "$RUN_ROOT" \
    --projection-report "$PROJECTION_REPORT" \
    --output "$SCRIPT_DIR/symmetry_projection_summary.csv"

echo
echo "Test completato."
echo "Invia: $SCRIPT_DIR/symmetry_projection_summary.csv"
echo "e, per quantificare i modi rimossi: $PROJECTION_REPORT"
