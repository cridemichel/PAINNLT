#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_TEMP_TRAIN_DIR:-dna_temporal_target_training}"
COPY_DIR="${DNA_SELF_FULL_RERUN_DIR:-dna_self_full_reruns}"
WINDOWS="${DNA_TEMP_TRAIN_WINDOWS:-1 2 5}"
EPOCHS="${DNA_TEMP_TRAIN_EPOCHS:-20}"
PATIENCE="${DNA_TEMP_TRAIN_PATIENCE:-6}"
BATCH="${DNA_TEMP_TRAIN_BATCH:-4}"
VAL_STRIDE="${DNA_TEMP_TRAIN_VALIDATION_STRIDE:-5}"
MARGIN="${DNA_TEMP_TRAIN_MARGIN_NM:-0.50}"
COMMON_MAX_WINDOW="${DNA_TEMP_TRAIN_COMMON_MAX_WINDOW_PS:-10}"
PREPARE_RERUNS="${DNA_TEMP_TRAIN_PREPARE_RERUNS:-0}"
REBUILD_TRAINER="${DNA_TEMP_TRAIN_REBUILD_TRAINER:-0}"
BUILD_JOBS="${DNA_TEMP_TRAIN_BUILD_JOBS:-4}"

SCRIPT_DIR="$(pwd)"
ROOT_DIR="$(cd ../.. && pwd)"
TRAIN_BUILD="$ROOT_DIR/training/build"
TRAINER="$TRAIN_BUILD/train_painn"
DATASET="$SCRIPT_DIR/tel22_dataset.bin"
BASE_CONFIG="$SCRIPT_DIR/tel22_training_config.json"
RAW_GRO="$SCRIPT_DIR/md.gro"
RAW_TRR="$SCRIPT_DIR/md.trr"
COPY_MANIFEST="$SCRIPT_DIR/$COPY_DIR/copy_groups.json"
TEMP_DIAG_REPORT="${DNA_TEMP_DIAG_REPORT:-$SCRIPT_DIR/dna_temporal_force_averaging/temporal_force_averaging_report.json}"
CACHE_DIR="$OUTDIR/target_cache"
TARGET_CACHE="$CACHE_DIR/temporal_self_targets.npz"
TARGET_CACHE_REPORT="$CACHE_DIR/temporal_self_targets_report.json"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in "$DATASET" "$BASE_CONFIG" "$RAW_GRO" "$RAW_TRR" \
         prepare_temporal_self_training_targets.py build_temporal_self_training_dataset.py \
         summarize_self_only_training.py summarize_temporal_target_training.py \
         03n_prepare_full_self_reruns.sh; do
    require_file "$f"
done
case "$EPOCHS" in ''|*[!0-9]*) echo "[ERROR] DNA_TEMP_TRAIN_EPOCHS deve essere intero" >&2; exit 2;; esac
case "$PATIENCE" in ''|*[!0-9]*) echo "[ERROR] DNA_TEMP_TRAIN_PATIENCE deve essere intero" >&2; exit 2;; esac
case "$BATCH" in ''|*[!0-9]*) echo "[ERROR] DNA_TEMP_TRAIN_BATCH deve essere intero" >&2; exit 2;; esac
case "$VAL_STRIDE" in ''|*[!0-9]*) echo "[ERROR] DNA_TEMP_TRAIN_VALIDATION_STRIDE deve essere intero" >&2; exit 2;; esac
[ "$EPOCHS" -gt 0 ] || { echo "[ERROR] epochs deve essere > 0" >&2; exit 2; }
[ "$PATIENCE" -gt 0 ] || { echo "[ERROR] patience deve essere > 0" >&2; exit 2; }
[ "$BATCH" -gt 0 ] || { echo "[ERROR] batch deve essere > 0" >&2; exit 2; }
[ "$VAL_STRIDE" -ge 2 ] || { echo "[ERROR] validation stride deve essere >= 2" >&2; exit 2; }

if [ "$PREPARE_RERUNS" = "1" ]; then
    DNA_SELF_FULL_RERUN_DIR="$COPY_DIR" bash 03n_prepare_full_self_reruns.sh
fi
require_file "$COPY_MANIFEST"
require_file "$SCRIPT_DIR/$COPY_DIR/full_self_rerun_report.json"
require_file "$TEMP_DIAG_REPORT"

if [ "$REBUILD_TRAINER" = "1" ]; then
    command -v cmake >/dev/null 2>&1 || { echo "[ERROR] cmake non trovato nel PATH." >&2; exit 1; }
    [ -d "$TRAIN_BUILD" ] || { echo "[ERROR] Directory build trainer non trovata: $TRAIN_BUILD" >&2; exit 1; }
    echo "[INFO] Rebuilding train_painn..."
    cmake --build "$TRAIN_BUILD" --target train_painn -j "$BUILD_JOBS"
fi
[ -x "$TRAINER" ] || { echo "[ERROR] Trainer non trovato/eseguibile: $TRAINER" >&2; exit 1; }

mkdir -p "$CACHE_DIR"
printf '%s\n' \
  "======================================================" \
  " 03q. TEL22 TEMPORAL TARGET TRAINING ABLATION" \
  "======================================================" \
  "Windows: $WINDOWS ps" \
  "Common center pool guard window: $COMMON_MAX_WINDOW ps (default matches the prior 1/2/5/10-ps diagnostic)." \
  "PaiNN: 64 x 2, 32 RBF | lr=0.001 | batch=$BATCH | torque_weight=0.5" \
  "Validation: identical deterministic temporal-stratified split, stride=$VAL_STRIDE" \
  "epochs=$EPOCHS | early_stopping_patience=$PATIENCE" \
  "No GROMACS reruns unless DNA_TEMP_TRAIN_PREPARE_RERUNS=1." \
  ""

# Build one shared target cache so all windows use exactly the same central frames.
"$PYTHON_BIN" prepare_temporal_self_training_targets.py \
    --dataset "$DATASET" \
    --raw-topology "$RAW_GRO" \
    --raw-trr "$RAW_TRR" \
    --copy-dir "$SCRIPT_DIR/$COPY_DIR" \
    --copy-manifest "$COPY_MANIFEST" \
    --window-ps $WINDOWS \
    --common-max-window-ps "$COMMON_MAX_WINDOW" \
    --output "$TARGET_CACHE" \
    --report "$TARGET_CACHE_REPORT"

$PYTHON_BIN - "$TARGET_CACHE_REPORT" "$TEMP_DIAG_REPORT" <<'PY'
import json,sys,math
a=json.load(open(sys.argv[1]))['inputs']
b=json.load(open(sys.argv[2]))['inputs']
checks=[
    ('common_center_frames', int(a['common_center_frames']), int(b['common_center_frames'])),
    ('common_center_time_start_ps', float(a['common_center_time_start_ps']), float(b['common_center_time_start_ps'])),
    ('common_center_time_end_ps', float(a['common_center_time_end_ps']), float(b['common_center_time_end_ps'])),
]
for name,x,y in checks:
    if isinstance(x, float):
        ok=math.isclose(x,y,abs_tol=1e-8,rel_tol=0.0)
    else:
        ok=(x==y)
    if not ok:
        raise SystemExit(f'matched temporal diagnostic mismatch for {name}: target-cache={x}, diagnostic={y}')
print('[INFO] Target cache and temporal diagnostic use the same common center-frame pool.')
PY

COMMON_FRAMES=$($PYTHON_BIN - "$TARGET_CACHE_REPORT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(int(r['inputs']['common_center_frames']))
PY
)
VAL_FRAMES=$((COMMON_FRAMES / VAL_STRIDE))
if [ "$VAL_FRAMES" -le 0 ] || [ "$VAL_FRAMES" -ge "$COMMON_FRAMES" ]; then
    echo "[ERROR] split non valido: common=$COMMON_FRAMES validation=$VAL_FRAMES" >&2
    exit 2
fi

echo "[INFO] Common center frames=$COMMON_FRAMES | expected train=$((COMMON_FRAMES-VAL_FRAMES)) val=$VAL_FRAMES"

for window in $WINDOWS; do
    KEY=$($PYTHON_BIN - "$window" <<'PY'
import sys
x=float(sys.argv[1])
print((f"{x:g}".replace('-', 'm').replace('.', 'p')) + 'ps')
PY
)
    CASE_DIR="$OUTDIR/$KEY"
    RUN_DIR="$CASE_DIR/run"
    DATA_BIN="$CASE_DIR/dataset.bin"
    DATA_REPORT="$CASE_DIR/dataset_report.json"
    RUN_CONFIG="$RUN_DIR/config.json"
    SUMMARY="$CASE_DIR/summary.json"
    rm -rf "$CASE_DIR"
    mkdir -p "$RUN_DIR"

    echo
    echo "[CASE] temporal target window=$window ps | key=$KEY"
    "$PYTHON_BIN" build_temporal_self_training_dataset.py \
        --dataset "$DATASET" \
        --config "$BASE_CONFIG" \
        --copy-manifest "$COPY_MANIFEST" \
        --target-cache "$TARGET_CACHE" \
        --target-cache-report "$TARGET_CACHE_REPORT" \
        --window-ps "$window" \
        --validation-stride "$VAL_STRIDE" \
        --margin "$MARGIN" \
        --output "$DATA_BIN" \
        --report "$DATA_REPORT"

    "$PYTHON_BIN" - "$BASE_CONFIG" "$RUN_CONFIG" "$DATA_REPORT" "$EPOCHS" "$BATCH" "$PATIENCE" <<'PY'
import json,sys
src,dst,report_path,epochs,batch,patience=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])
cfg=json.load(open(src))
rep=json.load(open(report_path))
val_frames=int(rep['split']['validation_frames'])
selected=int(rep['sampling']['selected_frames'])
if val_frames <= 0 or val_frames >= selected:
    raise SystemExit(f'invalid controlled split: selected={selected} validation={val_frames}')
cfg.update({
    'num_species': 8,
    'hidden_channels': 64,
    'n_layers': 2,
    'num_rbf': 32,
    'learning_rate': 0.001,
    'epochs': epochs,
    'batch_size': batch,
    'torque_weight': 0.5,
    'grad_clip_norm': 1.0,
    'weight_decay': 0.0,
    'lipschitz_lambda': 0.0,
    'diagnostic_overfit_frames': 0,
    'physical_validation_only': True,
    'include_decoys_in_train': False,
    'shuffle_each_epoch': True,
    'split_seed': 42,
    'validation_fraction': val_frames / selected,
    'validation_split_mode': 'tail',
    'validation_tail_frames': val_frames,
    'early_stopping_patience': patience,
    'reduce_lr_patience': min(6, patience),
    'report_grad_norms': True,
})
with open(dst,'w') as fh:
    json.dump(cfg,fh,indent=4)
    fh.write('\n')
PY

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    (
        cd "$RUN_DIR"
        "$TRAINER" "$SCRIPT_DIR/$DATA_BIN" model.pt config.json | tee run.log
    )
    if ! grep -q "Split mode: tail" "$RUN_DIR/run.log"; then
        echo "[ERROR] Il trainer eseguito non supporta lo split tail controllato." >&2
        echo "        Ricompila training/train_painn.cpp della patch 03n e rilancia 03q." >&2
        exit 3
    fi

    "$PYTHON_BIN" summarize_self_only_training.py \
        --training-csv "$RUN_DIR/cg_training_log.csv" \
        --dataset-report "$DATA_REPORT" \
        --output "$SUMMARY"
done

"$PYTHON_BIN" summarize_temporal_target_training.py \
    --root "$OUTDIR" \
    --temporal-report "$TEMP_DIAG_REPORT" \
    --output-json "$OUTDIR/temporal_target_training_summary.json" \
    --output-csv "$OUTDIR/temporal_target_training_summary.csv"

echo
echo "[DONE] Inviami soprattutto:"
echo "       $OUTDIR/temporal_target_training_summary.json"
echo "       $OUTDIR/temporal_target_training_summary.csv"
