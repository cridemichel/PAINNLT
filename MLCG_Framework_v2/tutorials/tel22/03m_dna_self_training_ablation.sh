#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_SELF_TRAIN_DIR:-dna_self_training_ablation}"
EPOCHS="${DNA_SELF_TRAIN_EPOCHS:-60}"
BATCH="${DNA_SELF_TRAIN_BATCH:-4}"
MARGIN="${DNA_SELF_TRAIN_MARGIN_NM:-0.50}"
COPY_DIR="${DNA_SELF_COPY_DIR:-dna_self_vs_intercopy}"

SCRIPT_DIR="$(pwd)"
ROOT_DIR="$(cd ../.. && pwd)"
TRAINER="$ROOT_DIR/training/build/train_painn"
DATASET="$SCRIPT_DIR/tel22_dataset.bin"
BASE_CONFIG="$SCRIPT_DIR/tel22_training_config.json"
RAW_GRO="$SCRIPT_DIR/md.gro"
RAW_TRR="$SCRIPT_DIR/md.trr"
COPY_MANIFEST="$SCRIPT_DIR/$COPY_DIR/copy_groups.json"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in "$DATASET" "$BASE_CONFIG" "$RAW_GRO" "$RAW_TRR" "$COPY_MANIFEST" \
         build_dna_self_isolated_dataset.py summarize_self_only_training.py; do
    require_file "$f"
done
[ -x "$TRAINER" ] || { echo "[ERROR] Trainer non trovato/eseguibile: $TRAINER" >&2; exit 1; }

case "$EPOCHS" in ''|*[!0-9]*) echo "[ERROR] DNA_SELF_TRAIN_EPOCHS deve essere intero" >&2; exit 2;; esac
case "$BATCH" in ''|*[!0-9]*) echo "[ERROR] DNA_SELF_TRAIN_BATCH deve essere intero" >&2; exit 2;; esac
[ "$EPOCHS" -gt 0 ] || { echo "[ERROR] epochs deve essere > 0" >&2; exit 2; }
[ "$BATCH" -gt 0 ] || { echo "[ERROR] batch deve essere > 0" >&2; exit 2; }

NCOPIES=$($PYTHON_BIN - "$COPY_MANIFEST" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['copies']))
PY
)
for ((ci=0; ci<NCOPIES; ci++)); do
    tag=$(printf 'copy_%02d' "$ci")
    require_file "$SCRIPT_DIR/$COPY_DIR/${tag}.gro"
    require_file "$SCRIPT_DIR/$COPY_DIR/${tag}_rerun.trr"
done

mkdir -p "$OUTDIR"
SELF_DATASET="$OUTDIR/tel22_self_isolated.bin"
DATASET_REPORT="$OUTDIR/tel22_self_isolated_report.json"
RUN_DIR="$OUTDIR/run"
RUN_CONFIG="$RUN_DIR/config.json"
SUMMARY="$OUTDIR/self_training_summary.json"

printf '%s\n' \
  "======================================================" \
  " 03m. TEL22 DNA SELF-ONLY TRAINING ABLATION" \
  "======================================================" \
  "Target: single-copy DNA self force/torque from 03l reruns" \
  "Input: same CG copies translated apart so no inter-copy PaiNN edge survives" \
  "Architecture: 64 x 2, 32 RBF, cutoff inherited from tel22_training_config.json" \
  "epochs=$EPOCHS | batch=$BATCH | torque_weight=0.5" \
  "Diagnostic only: production preprocessing/priors are not modified." \
  ""

"$PYTHON_BIN" build_dna_self_isolated_dataset.py \
    --dataset "$DATASET" \
    --config "$BASE_CONFIG" \
    --raw-topology "$RAW_GRO" \
    --raw-trr "$RAW_TRR" \
    --copy-dir "$SCRIPT_DIR/$COPY_DIR" \
    --copy-manifest "$COPY_MANIFEST" \
    --margin "$MARGIN" \
    --output "$SELF_DATASET" \
    --report "$DATASET_REPORT"

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" - "$BASE_CONFIG" "$RUN_CONFIG" "$EPOCHS" "$BATCH" <<'PY'
import json,sys
src,dst,epochs,batch=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
cfg=json.load(open(src))
cfg.update({
    "num_species": 8,
    "hidden_channels": 64,
    "n_layers": 2,
    "num_rbf": 32,
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
    "early_stopping_patience": max(20, epochs),
    "reduce_lr_patience": 8,
    "report_grad_norms": True,
})
with open(dst,'w') as fh:
    json.dump(cfg,fh,indent=4)
    fh.write('\n')
PY

export PYTORCH_ENABLE_MPS_FALLBACK=1
(
    cd "$RUN_DIR"
    "$TRAINER" "$SCRIPT_DIR/$SELF_DATASET" model.pt config.json | tee run.log
)

"$PYTHON_BIN" summarize_self_only_training.py \
    --training-csv "$RUN_DIR/cg_training_log.csv" \
    --dataset-report "$DATASET_REPORT" \
    --conditional-report "$SCRIPT_DIR/$COPY_DIR/dna_self_vs_intercopy_report.json" \
    --output "$SUMMARY"

echo
echo "[DONE] Inviami soprattutto:"
echo "       $SUMMARY"
echo "       $RUN_DIR/cg_training_log.csv"
