#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_SELF_LC_DIR:-dna_self_learning_curve}"
COPY_DIR="${DNA_SELF_FULL_RERUN_DIR:-dna_self_full_reruns}"
SIZES="${DNA_SELF_LC_SIZES:-101 501 1001}"
MODES="${DNA_SELF_LC_TARGETS:-total}"
EPOCHS="${DNA_SELF_LC_EPOCHS:-20}"
PATIENCE="${DNA_SELF_LC_PATIENCE:-6}"
BATCH="${DNA_SELF_LC_BATCH:-4}"
VAL_STRIDE="${DNA_SELF_LC_VALIDATION_STRIDE:-5}"
MARGIN="${DNA_SELF_LC_MARGIN_NM:-0.50}"
PREPARE="${DNA_SELF_LC_PREPARE:-1}"
REBUILD_TRAINER="${DNA_SELF_LC_REBUILD_TRAINER:-1}"
BUILD_JOBS="${DNA_SELF_LC_BUILD_JOBS:-4}"

SCRIPT_DIR="$(pwd)"
ROOT_DIR="$(cd ../.. && pwd)"
TRAIN_BUILD="$ROOT_DIR/training/build"
TRAINER="$TRAIN_BUILD/train_painn"
DATASET="$SCRIPT_DIR/tel22_dataset.bin"
BASE_CONFIG="$SCRIPT_DIR/tel22_training_config.json"
RAW_GRO="$SCRIPT_DIR/md.gro"
RAW_TRR="$SCRIPT_DIR/md.trr"
PRIORS="$SCRIPT_DIR/cg_priors.json"
COPY_MANIFEST="$SCRIPT_DIR/$COPY_DIR/copy_groups.json"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in "$DATASET" "$BASE_CONFIG" "$RAW_GRO" "$RAW_TRR" "$PRIORS" \
         build_dna_self_isolated_dataset.py summarize_self_only_training.py summarize_self_learning_curve.py \
         03n_prepare_full_self_reruns.sh; do
    require_file "$f"
done
case "$EPOCHS" in ''|*[!0-9]*) echo "[ERROR] DNA_SELF_LC_EPOCHS deve essere intero" >&2; exit 2;; esac
case "$PATIENCE" in ''|*[!0-9]*) echo "[ERROR] DNA_SELF_LC_PATIENCE deve essere intero" >&2; exit 2;; esac
case "$BATCH" in ''|*[!0-9]*) echo "[ERROR] DNA_SELF_LC_BATCH deve essere intero" >&2; exit 2;; esac
case "$VAL_STRIDE" in ''|*[!0-9]*) echo "[ERROR] DNA_SELF_LC_VALIDATION_STRIDE deve essere intero" >&2; exit 2;; esac
[ "$EPOCHS" -gt 0 ] || { echo "[ERROR] epochs deve essere > 0" >&2; exit 2; }
[ "$PATIENCE" -gt 0 ] || { echo "[ERROR] patience deve essere > 0" >&2; exit 2; }
[ "$BATCH" -gt 0 ] || { echo "[ERROR] batch deve essere > 0" >&2; exit 2; }
[ "$VAL_STRIDE" -ge 2 ] || { echo "[ERROR] validation stride deve essere >= 2" >&2; exit 2; }

if [ "$PREPARE" = "1" ]; then
    DNA_SELF_FULL_RERUN_DIR="$COPY_DIR" bash 03n_prepare_full_self_reruns.sh
fi
require_file "$COPY_MANIFEST"
require_file "$SCRIPT_DIR/$COPY_DIR/full_self_rerun_report.json"

AVAILABLE=$($PYTHON_BIN - "$SCRIPT_DIR/$COPY_DIR/full_self_rerun_report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(int(r['frames_per_copy'][0]))
PY
)

if [ "$REBUILD_TRAINER" = "1" ]; then
    command -v cmake >/dev/null 2>&1 || { echo "[ERROR] cmake non trovato nel PATH." >&2; exit 1; }
    [ -d "$TRAIN_BUILD" ] || { echo "[ERROR] Directory build trainer non trovata: $TRAIN_BUILD" >&2; exit 1; }
    echo "[INFO] Rebuilding train_painn per abilitare validation_split_mode=tail..."
    cmake --build "$TRAIN_BUILD" --target train_painn -j "$BUILD_JOBS"
fi
[ -x "$TRAINER" ] || { echo "[ERROR] Trainer non trovato/eseguibile: $TRAINER" >&2; exit 1; }

mkdir -p "$OUTDIR"
printf '%s\n' \
  "======================================================" \
  " 03n. TEL22 DNA SELF LEARNING CURVE" \
  "======================================================" \
  "Available full self rerun frames: $AVAILABLE" \
  "Sizes:   $SIZES" \
  "Targets: $MODES" \
  "PaiNN: 64 x 2, 32 RBF | lr=0.001 | batch=$BATCH | torque_weight=0.5" \
  "Validation: deterministic temporal-stratified every $VAL_STRIDE-th selected frame" \
  "epochs=$EPOCHS with early_stopping_patience=$PATIENCE" \
  ""

for mode in $MODES; do
    case "$mode" in total|residual) ;; *) echo "[ERROR] Target mode non valido: $mode" >&2; exit 2;; esac
    for n in $SIZES; do
        case "$n" in ''|*[!0-9]*) echo "[ERROR] Sample size non intera: $n" >&2; exit 2;; esac
        if [ "$n" -gt "$AVAILABLE" ]; then
            echo "[WARN] Skip target=$mode N=$n: disponibili solo $AVAILABLE frame."
            continue
        fi
        if [ "$n" -lt "$VAL_STRIDE" ]; then
            echo "[WARN] Skip target=$mode N=$n: servono almeno $VAL_STRIDE frame per lo split."
            continue
        fi

        CASE_DIR="$OUTDIR/$mode/$n"
        RUN_DIR="$CASE_DIR/run"
        DATA_BIN="$CASE_DIR/dataset.bin"
        DATA_REPORT="$CASE_DIR/dataset_report.json"
        RUN_CONFIG="$RUN_DIR/config.json"
        SUMMARY="$CASE_DIR/summary.json"
        rm -rf "$CASE_DIR"
        mkdir -p "$RUN_DIR"

        echo
        echo "[CASE] target=$mode | selected frames=$n"
        builder_args=(
          --dataset "$DATASET"
          --config "$BASE_CONFIG"
          --raw-topology "$RAW_GRO"
          --raw-trr "$RAW_TRR"
          --copy-dir "$SCRIPT_DIR/$COPY_DIR"
          --copy-manifest "$COPY_MANIFEST"
          --target-mode "$mode"
          --sample-count "$n"
          --validation-stride "$VAL_STRIDE"
          --margin "$MARGIN"
          --output "$DATA_BIN"
          --report "$DATA_REPORT"
        )
        if [ "$mode" = "residual" ]; then
            builder_args+=(--priors "$PRIORS")
        fi
        "$PYTHON_BIN" build_dna_self_isolated_dataset.py "${builder_args[@]}"

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
    'reduce_lr_patience': 6,
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
            echo "        Ricompila training/train_painn.cpp della patch e rilancia 03n." >&2
            exit 3
        fi

        summary_args=(
          --training-csv "$RUN_DIR/cg_training_log.csv"
          --dataset-report "$DATA_REPORT"
          --output "$SUMMARY"
        )
        if [ -f "$SCRIPT_DIR/dna_self_vs_intercopy/dna_self_vs_intercopy_report.json" ]; then
            summary_args+=(--conditional-report "$SCRIPT_DIR/dna_self_vs_intercopy/dna_self_vs_intercopy_report.json")
        fi
        "$PYTHON_BIN" summarize_self_only_training.py "${summary_args[@]}"
    done
done

"$PYTHON_BIN" summarize_self_learning_curve.py \
    --root "$OUTDIR" \
    --output-json "$OUTDIR/self_learning_curve_summary.json" \
    --output-csv "$OUTDIR/self_learning_curve_summary.csv"

echo
echo "[DONE] Inviami soprattutto:"
echo "       $OUTDIR/self_learning_curve_summary.json"
echo "       $OUTDIR/self_learning_curve_summary.csv"
