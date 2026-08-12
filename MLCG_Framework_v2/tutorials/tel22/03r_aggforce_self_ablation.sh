#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_AGGFORCE_DIR:-dna_aggforce_self_ablation}"
COPY_DIR="${DNA_SELF_FULL_RERUN_DIR:-dna_self_full_reruns}"
TRAIN_VARIANTS="${DNA_AGGFORCE_TRAIN_VARIANTS:-optimized}"
EPOCHS="${DNA_AGGFORCE_EPOCHS:-20}"
PATIENCE="${DNA_AGGFORCE_PATIENCE:-6}"
BATCH="${DNA_AGGFORCE_BATCH:-4}"
VAL_STRIDE="${DNA_AGGFORCE_VALIDATION_STRIDE:-5}"
MARGIN="${DNA_AGGFORCE_MARGIN_NM:-0.50}"
FIT_MAX_SAMPLES="${DNA_AGGFORCE_FIT_MAX_SAMPLES:-3000}"
CONSTRAINT_FRAMES="${DNA_AGGFORCE_CONSTRAINT_FRAMES:-100}"
CONSTRAINT_THRESHOLD="${DNA_AGGFORCE_CONSTRAINT_THRESHOLD:-0.001}"
L2="${DNA_AGGFORCE_L2:-1000.0}"
DIAGNOSTIC_ONLY="${DNA_AGGFORCE_DIAGNOSTIC_ONLY:-0}"
REUSE_PREPARED="${DNA_AGGFORCE_REUSE_PREPARED:-0}"
REBUILD_TRAINER="${DNA_AGGFORCE_REBUILD_TRAINER:-0}"
BUILD_JOBS="${DNA_AGGFORCE_BUILD_JOBS:-4}"

SCRIPT_DIR="$(pwd)"
ROOT_DIR="$(cd ../.. && pwd)"
TRAIN_BUILD="$ROOT_DIR/training/build"
TRAINER="$TRAIN_BUILD/train_painn"
DATASET="$SCRIPT_DIR/tel22_dataset.bin"
BASE_CONFIG="$SCRIPT_DIR/tel22_training_config.json"
RAW_GRO="$SCRIPT_DIR/md.gro"
RAW_TRR="$SCRIPT_DIR/md.trr"
MAPPING_CONFIG="$SCRIPT_DIR/tel22_topology.json"
COPY_MANIFEST="$SCRIPT_DIR/$COPY_DIR/copy_groups.json"
TEMP_REPORT="${DNA_AGGFORCE_TEMP_REPORT:-$SCRIPT_DIR/dna_temporal_force_averaging/temporal_force_averaging_report.json}"
TEMP_PAIRS="${DNA_AGGFORCE_TEMP_PAIRS:-$SCRIPT_DIR/dna_temporal_force_averaging/temporal_force_averaging_pairs.csv}"
PRIOR_TEMP_TRAIN="${DNA_AGGFORCE_PRIOR_TEMP_TRAIN:-$SCRIPT_DIR/dna_temporal_target_training/temporal_target_training_summary.json}"
CACHE_DIR="$OUTDIR/force_map"
TARGET_CACHE="$CACHE_DIR/aggforce_self_targets.npz"
MAPS_NPZ="$CACHE_DIR/aggforce_maps.npz"
MAP_REPORT="$CACHE_DIR/aggforce_force_mapping_report.json"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in "$DATASET" "$BASE_CONFIG" "$RAW_GRO" "$RAW_TRR" "$MAPPING_CONFIG" "$COPY_MANIFEST" "$TEMP_REPORT" "$TEMP_PAIRS" \
         prepare_aggforce_self_targets.py build_aggforce_self_training_dataset.py summarize_aggforce_self_ablation.py; do
    require_file "$f"
done
case "$EPOCHS" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_EPOCHS deve essere intero" >&2; exit 2;; esac
case "$PATIENCE" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_PATIENCE deve essere intero" >&2; exit 2;; esac
case "$BATCH" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_BATCH deve essere intero" >&2; exit 2;; esac
case "$VAL_STRIDE" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_VALIDATION_STRIDE deve essere intero" >&2; exit 2;; esac
case "$CONSTRAINT_FRAMES" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_CONSTRAINT_FRAMES deve essere intero" >&2; exit 2;; esac
case "$DIAGNOSTIC_ONLY" in 0|1) ;; *) echo "[ERROR] DNA_AGGFORCE_DIAGNOSTIC_ONLY deve essere 0 o 1" >&2; exit 2;; esac
case "$REUSE_PREPARED" in 0|1) ;; *) echo "[ERROR] DNA_AGGFORCE_REUSE_PREPARED deve essere 0 o 1" >&2; exit 2;; esac
[ "$EPOCHS" -gt 0 ] && [ "$PATIENCE" -gt 0 ] && [ "$BATCH" -gt 0 ] || { echo "[ERROR] epochs/patience/batch devono essere >0" >&2; exit 2; }
[ "$VAL_STRIDE" -ge 2 ] || { echo "[ERROR] validation stride deve essere >=2" >&2; exit 2; }
[ "$CONSTRAINT_FRAMES" -gt 0 ] || { echo "[ERROR] constraint frames deve essere >0" >&2; exit 2; }

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import aggforce
from aggforce import LinearMap, constraint_aware_uni_map, guess_pairwise_constraints, project_forces, qp_linear_map
PY
then
    cat >&2 <<'EOF'
[ERROR] aggforce non e' installato (o manca l'API linear-map richiesta).
Installa il repository ufficiale nello stesso ambiente Python usato dal tutorial:

  python3 -m pip install 'git+https://github.com/noegroup/aggforce.git'

Poi rilancia 03r_aggforce_self_ablation.sh.
EOF
    exit 4
fi

mkdir -p "$CACHE_DIR"
printf '%s\n' \
  "======================================================" \
  " 03r. TEL22 STATISTICALLY OPTIMAL FORCE AGGREGATION" \
  "======================================================" \
  "Coordinate mapping: unchanged" \
  "aggforce scope: exact single-site COM residues only (DA/DT); multi-site rigid DG unchanged" \
  "Force maps diagnosed: current residue-sum / constraint-aware DA-DT / qp-optimized DA-DT" \
  "aggforce fit pool: TRAINING CENTER FRAMES ONLY; max samples=$FIT_MAX_SAMPLES" \
  "constraints: first $CONSTRAINT_FRAMES fit samples, threshold=$CONSTRAINT_THRESHOLD" \
  "qp l2_regularization=$L2" \
  "Training variants: $TRAIN_VARIANTS" \
  "PaiNN: 64 x 2, 32 RBF | lr=0.001 | batch=$BATCH | torque_weight=0.5 (same as 03q)" \
  "epochs=$EPOCHS | early_stopping_patience=$PATIENCE" \
  ""

if [ "$REUSE_PREPARED" = "1" ] && [ -f "$TARGET_CACHE" ] && [ -f "$MAPS_NPZ" ] && [ -f "$MAP_REPORT" ]; then
    echo "[REUSE] Riutilizzo target/mappe aggforce gia' preparati in $CACHE_DIR"
else
    "$PYTHON_BIN" prepare_aggforce_self_targets.py \
        --dataset "$DATASET" \
        --mapping-config "$MAPPING_CONFIG" \
        --raw-topology "$RAW_GRO" \
        --raw-trr "$RAW_TRR" \
        --copy-dir "$SCRIPT_DIR/$COPY_DIR" \
        --copy-manifest "$COPY_MANIFEST" \
        --temporal-report "$TEMP_REPORT" \
        --temporal-pairs "$TEMP_PAIRS" \
        --validation-stride "$VAL_STRIDE" \
        --constraint-threshold "$CONSTRAINT_THRESHOLD" \
        --constraint-frames "$CONSTRAINT_FRAMES" \
        --fit-max-samples "$FIT_MAX_SAMPLES" \
        --l2-regularization "$L2" \
        --output-cache "$TARGET_CACHE" \
        --output-maps "$MAPS_NPZ" \
        --output-report "$MAP_REPORT"
fi

if [ "$DIAGNOSTIC_ONLY" = "1" ]; then
    echo "[DONE] Diagnostic-only mode. Inviami: $MAP_REPORT"
    exit 0
fi

if [ "$REBUILD_TRAINER" = "1" ]; then
    command -v cmake >/dev/null 2>&1 || { echo "[ERROR] cmake non trovato nel PATH." >&2; exit 1; }
    [ -d "$TRAIN_BUILD" ] || { echo "[ERROR] Directory build trainer non trovata: $TRAIN_BUILD" >&2; exit 1; }
    cmake --build "$TRAIN_BUILD" --target train_painn -j "$BUILD_JOBS"
fi
[ -x "$TRAINER" ] || { echo "[ERROR] Trainer non trovato/eseguibile: $TRAINER" >&2; exit 1; }

for variant in $TRAIN_VARIANTS; do
    case "$variant" in current|constraint_aware|optimized) ;; *) echo "[ERROR] variante aggforce sconosciuta: $variant" >&2; exit 2;; esac
    CASE_DIR="$OUTDIR/$variant"
    RUN_DIR="$CASE_DIR/run"
    DATA_BIN="$CASE_DIR/dataset.bin"
    DATA_REPORT="$CASE_DIR/dataset_report.json"
    RUN_CONFIG="$RUN_DIR/config.json"
    rm -rf "$CASE_DIR"
    mkdir -p "$RUN_DIR"

    echo
    echo "[CASE] force map = $variant"
    "$PYTHON_BIN" build_aggforce_self_training_dataset.py \
        --dataset "$DATASET" \
        --config "$BASE_CONFIG" \
        --copy-manifest "$COPY_MANIFEST" \
        --target-cache "$TARGET_CACHE" \
        --aggforce-report "$MAP_REPORT" \
        --variant "$variant" \
        --margin "$MARGIN" \
        --output "$DATA_BIN" \
        --report "$DATA_REPORT"

    "$PYTHON_BIN" - "$BASE_CONFIG" "$RUN_CONFIG" "$DATA_REPORT" "$EPOCHS" "$BATCH" "$PATIENCE" <<'PY'
import json,sys
src,dst,report_path,epochs,batch,patience=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])
cfg=json.load(open(src)); rep=json.load(open(report_path))
val_frames=int(rep['split']['validation_frames']); selected=int(rep['sampling']['selected_frames'])
if val_frames <= 0 or val_frames >= selected:
    raise SystemExit(f'invalid controlled split: selected={selected} validation={val_frames}')
cfg.update({
    'num_species':8,
    'hidden_channels':64,
    'n_layers':2,
    'num_rbf':32,
    'learning_rate':0.001,
    'epochs':epochs,
    'batch_size':batch,
    'torque_weight':0.5,
    'grad_clip_norm':1.0,
    'weight_decay':0.0,
    'lipschitz_lambda':0.0,
    'diagnostic_overfit_frames':0,
    'physical_validation_only':True,
    'include_decoys_in_train':False,
    'shuffle_each_epoch':True,
    'split_seed':42,
    'validation_fraction':val_frames/selected,
    'validation_split_mode':'tail',
    'validation_tail_frames':val_frames,
    'early_stopping_patience':patience,
    'reduce_lr_patience':min(6,patience),
    'report_grad_norms':True,
})
with open(dst,'w') as fh:
    json.dump(cfg,fh,indent=4); fh.write('\n')
PY

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    (
        cd "$RUN_DIR"
        "$TRAINER" "$SCRIPT_DIR/$DATA_BIN" model.pt config.json | tee run.log
    )
    if ! grep -q "Split mode: tail" "$RUN_DIR/run.log"; then
        echo "[ERROR] Il trainer non supporta lo split tail controllato; ricompila la patch 03n." >&2
        exit 3
    fi
done

SUMMARY_ARGS=(
    --root "$OUTDIR"
    --aggforce-report "$MAP_REPORT"
    --output-json "$OUTDIR/aggforce_training_summary.json"
    --output-csv "$OUTDIR/aggforce_training_summary.csv"
)
if [ -f "$PRIOR_TEMP_TRAIN" ]; then
    SUMMARY_ARGS+=(--temporal-training-summary "$PRIOR_TEMP_TRAIN")
fi
"$PYTHON_BIN" summarize_aggforce_self_ablation.py "${SUMMARY_ARGS[@]}"

echo
echo "[DONE] Inviami soprattutto:"
echo "       $MAP_REPORT"
echo "       $OUTDIR/aggforce_training_summary.json"
echo "       $OUTDIR/aggforce_training_summary.csv"
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_AGGFORCE_DIR:-dna_aggforce_self_ablation}"
COPY_DIR="${DNA_SELF_FULL_RERUN_DIR:-dna_self_full_reruns}"
TRAIN_VARIANTS="${DNA_AGGFORCE_TRAIN_VARIANTS:-optimized}"
EPOCHS="${DNA_AGGFORCE_EPOCHS:-20}"
PATIENCE="${DNA_AGGFORCE_PATIENCE:-6}"
BATCH="${DNA_AGGFORCE_BATCH:-4}"
VAL_STRIDE="${DNA_AGGFORCE_VALIDATION_STRIDE:-5}"
MARGIN="${DNA_AGGFORCE_MARGIN_NM:-0.50}"
FIT_MAX_SAMPLES="${DNA_AGGFORCE_FIT_MAX_SAMPLES:-3000}"
CONSTRAINT_FRAMES="${DNA_AGGFORCE_CONSTRAINT_FRAMES:-100}"
CONSTRAINT_THRESHOLD="${DNA_AGGFORCE_CONSTRAINT_THRESHOLD:-0.001}"
L2="${DNA_AGGFORCE_L2:-1000.0}"
DIAGNOSTIC_ONLY="${DNA_AGGFORCE_DIAGNOSTIC_ONLY:-0}"
REUSE_PREPARED="${DNA_AGGFORCE_REUSE_PREPARED:-0}"
REBUILD_TRAINER="${DNA_AGGFORCE_REBUILD_TRAINER:-0}"
BUILD_JOBS="${DNA_AGGFORCE_BUILD_JOBS:-4}"

SCRIPT_DIR="$(pwd)"
ROOT_DIR="$(cd ../.. && pwd)"
TRAIN_BUILD="$ROOT_DIR/training/build"
TRAINER="$TRAIN_BUILD/train_painn"
DATASET="$SCRIPT_DIR/tel22_dataset.bin"
BASE_CONFIG="$SCRIPT_DIR/tel22_training_config.json"
RAW_GRO="$SCRIPT_DIR/md.gro"
RAW_TRR="$SCRIPT_DIR/md.trr"
MAPPING_CONFIG="$SCRIPT_DIR/tel22_topology.json"
COPY_MANIFEST="$SCRIPT_DIR/$COPY_DIR/copy_groups.json"
TEMP_REPORT="${DNA_AGGFORCE_TEMP_REPORT:-$SCRIPT_DIR/dna_temporal_force_averaging/temporal_force_averaging_report.json}"
TEMP_PAIRS="${DNA_AGGFORCE_TEMP_PAIRS:-$SCRIPT_DIR/dna_temporal_force_averaging/temporal_force_averaging_pairs.csv}"
PRIOR_TEMP_TRAIN="${DNA_AGGFORCE_PRIOR_TEMP_TRAIN:-$SCRIPT_DIR/dna_temporal_target_training/temporal_target_training_summary.json}"
CACHE_DIR="$OUTDIR/force_map"
TARGET_CACHE="$CACHE_DIR/aggforce_self_targets.npz"
MAPS_NPZ="$CACHE_DIR/aggforce_maps.npz"
MAP_REPORT="$CACHE_DIR/aggforce_force_mapping_report.json"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in "$DATASET" "$BASE_CONFIG" "$RAW_GRO" "$RAW_TRR" "$MAPPING_CONFIG" "$COPY_MANIFEST" "$TEMP_REPORT" "$TEMP_PAIRS" \
         prepare_aggforce_self_targets.py build_aggforce_self_training_dataset.py summarize_aggforce_self_ablation.py; do
    require_file "$f"
done
case "$EPOCHS" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_EPOCHS deve essere intero" >&2; exit 2;; esac
case "$PATIENCE" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_PATIENCE deve essere intero" >&2; exit 2;; esac
case "$BATCH" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_BATCH deve essere intero" >&2; exit 2;; esac
case "$VAL_STRIDE" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_VALIDATION_STRIDE deve essere intero" >&2; exit 2;; esac
case "$CONSTRAINT_FRAMES" in ''|*[!0-9]*) echo "[ERROR] DNA_AGGFORCE_CONSTRAINT_FRAMES deve essere intero" >&2; exit 2;; esac
case "$DIAGNOSTIC_ONLY" in 0|1) ;; *) echo "[ERROR] DNA_AGGFORCE_DIAGNOSTIC_ONLY deve essere 0 o 1" >&2; exit 2;; esac
case "$REUSE_PREPARED" in 0|1) ;; *) echo "[ERROR] DNA_AGGFORCE_REUSE_PREPARED deve essere 0 o 1" >&2; exit 2;; esac
[ "$EPOCHS" -gt 0 ] && [ "$PATIENCE" -gt 0 ] && [ "$BATCH" -gt 0 ] || { echo "[ERROR] epochs/patience/batch devono essere >0" >&2; exit 2; }
[ "$VAL_STRIDE" -ge 2 ] || { echo "[ERROR] validation stride deve essere >=2" >&2; exit 2; }
[ "$CONSTRAINT_FRAMES" -gt 0 ] || { echo "[ERROR] constraint frames deve essere >0" >&2; exit 2; }

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import aggforce
from aggforce import LinearMap, constraint_aware_uni_map, guess_pairwise_constraints, project_forces, qp_linear_map
PY
then
    cat >&2 <<'EOF'
[ERROR] aggforce non e' installato (o manca l'API linear-map richiesta).
Installa il repository ufficiale nello stesso ambiente Python usato dal tutorial:

  python3 -m pip install 'git+https://github.com/noegroup/aggforce.git'

Poi rilancia 03r_aggforce_self_ablation.sh.
EOF
    exit 4
fi

mkdir -p "$CACHE_DIR"
printf '%s\n' \
  "======================================================" \
  " 03r. TEL22 STATISTICALLY OPTIMAL FORCE AGGREGATION" \
  "======================================================" \
  "Coordinate mapping: unchanged" \
  "aggforce scope: exact single-site COM residues only (DA/DT); multi-site rigid DG unchanged" \
  "Force maps diagnosed: current residue-sum / constraint-aware DA-DT / qp-optimized DA-DT" \
  "aggforce fit pool: TRAINING CENTER FRAMES ONLY; max samples=$FIT_MAX_SAMPLES" \
  "constraints: first $CONSTRAINT_FRAMES fit samples, threshold=$CONSTRAINT_THRESHOLD" \
  "qp l2_regularization=$L2" \
  "Training variants: $TRAIN_VARIANTS" \
  "PaiNN: 64 x 2, 32 RBF | lr=0.001 | batch=$BATCH | torque_weight=0.5 (same as 03q)" \
  "epochs=$EPOCHS | early_stopping_patience=$PATIENCE" \
  ""

if [ "$REUSE_PREPARED" = "1" ] && [ -f "$TARGET_CACHE" ] && [ -f "$MAPS_NPZ" ] && [ -f "$MAP_REPORT" ]; then
    echo "[REUSE] Riutilizzo target/mappe aggforce gia' preparati in $CACHE_DIR"
else
    "$PYTHON_BIN" prepare_aggforce_self_targets.py \
        --dataset "$DATASET" \
        --mapping-config "$MAPPING_CONFIG" \
        --raw-topology "$RAW_GRO" \
        --raw-trr "$RAW_TRR" \
        --copy-dir "$SCRIPT_DIR/$COPY_DIR" \
        --copy-manifest "$COPY_MANIFEST" \
        --temporal-report "$TEMP_REPORT" \
        --temporal-pairs "$TEMP_PAIRS" \
        --validation-stride "$VAL_STRIDE" \
        --constraint-threshold "$CONSTRAINT_THRESHOLD" \
        --constraint-frames "$CONSTRAINT_FRAMES" \
        --fit-max-samples "$FIT_MAX_SAMPLES" \
        --l2-regularization "$L2" \
        --output-cache "$TARGET_CACHE" \
        --output-maps "$MAPS_NPZ" \
        --output-report "$MAP_REPORT"
fi

if [ "$DIAGNOSTIC_ONLY" = "1" ]; then
    echo "[DONE] Diagnostic-only mode. Inviami: $MAP_REPORT"
    exit 0
fi

if [ "$REBUILD_TRAINER" = "1" ]; then
    command -v cmake >/dev/null 2>&1 || { echo "[ERROR] cmake non trovato nel PATH." >&2; exit 1; }
    [ -d "$TRAIN_BUILD" ] || { echo "[ERROR] Directory build trainer non trovata: $TRAIN_BUILD" >&2; exit 1; }
    cmake --build "$TRAIN_BUILD" --target train_painn -j "$BUILD_JOBS"
fi
[ -x "$TRAINER" ] || { echo "[ERROR] Trainer non trovato/eseguibile: $TRAINER" >&2; exit 1; }

for variant in $TRAIN_VARIANTS; do
    case "$variant" in current|constraint_aware|optimized) ;; *) echo "[ERROR] variante aggforce sconosciuta: $variant" >&2; exit 2;; esac
    CASE_DIR="$OUTDIR/$variant"
    RUN_DIR="$CASE_DIR/run"
    DATA_BIN="$CASE_DIR/dataset.bin"
    DATA_REPORT="$CASE_DIR/dataset_report.json"
    RUN_CONFIG="$RUN_DIR/config.json"
    rm -rf "$CASE_DIR"
    mkdir -p "$RUN_DIR"

    echo
    echo "[CASE] force map = $variant"
    "$PYTHON_BIN" build_aggforce_self_training_dataset.py \
        --dataset "$DATASET" \
        --config "$BASE_CONFIG" \
        --copy-manifest "$COPY_MANIFEST" \
        --target-cache "$TARGET_CACHE" \
        --aggforce-report "$MAP_REPORT" \
        --variant "$variant" \
        --margin "$MARGIN" \
        --output "$DATA_BIN" \
        --report "$DATA_REPORT"

    "$PYTHON_BIN" - "$BASE_CONFIG" "$RUN_CONFIG" "$DATA_REPORT" "$EPOCHS" "$BATCH" "$PATIENCE" <<'PY'
import json,sys
src,dst,report_path,epochs,batch,patience=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])
cfg=json.load(open(src)); rep=json.load(open(report_path))
val_frames=int(rep['split']['validation_frames']); selected=int(rep['sampling']['selected_frames'])
if val_frames <= 0 or val_frames >= selected:
    raise SystemExit(f'invalid controlled split: selected={selected} validation={val_frames}')
cfg.update({
    'num_species':8,
    'hidden_channels':64,
    'n_layers':2,
    'num_rbf':32,
    'learning_rate':0.001,
    'epochs':epochs,
    'batch_size':batch,
    'torque_weight':0.5,
    'grad_clip_norm':1.0,
    'weight_decay':0.0,
    'lipschitz_lambda':0.0,
    'diagnostic_overfit_frames':0,
    'physical_validation_only':True,
    'include_decoys_in_train':False,
    'shuffle_each_epoch':True,
    'split_seed':42,
    'validation_fraction':val_frames/selected,
    'validation_split_mode':'tail',
    'validation_tail_frames':val_frames,
    'early_stopping_patience':patience,
    'reduce_lr_patience':min(6,patience),
    'report_grad_norms':True,
})
with open(dst,'w') as fh:
    json.dump(cfg,fh,indent=4); fh.write('\n')
PY

    export PYTORCH_ENABLE_MPS_FALLBACK=1
    (
        cd "$RUN_DIR"
        "$TRAINER" "$SCRIPT_DIR/$DATA_BIN" model.pt config.json | tee run.log
    )
    if ! grep -q "Split mode: tail" "$RUN_DIR/run.log"; then
        echo "[ERROR] Il trainer non supporta lo split tail controllato; ricompila la patch 03n." >&2
        exit 3
    fi
done

SUMMARY_ARGS=(
    --root "$OUTDIR"
    --aggforce-report "$MAP_REPORT"
    --output-json "$OUTDIR/aggforce_training_summary.json"
    --output-csv "$OUTDIR/aggforce_training_summary.csv"
)
if [ -f "$PRIOR_TEMP_TRAIN" ]; then
    SUMMARY_ARGS+=(--temporal-training-summary "$PRIOR_TEMP_TRAIN")
fi
"$PYTHON_BIN" summarize_aggforce_self_ablation.py "${SUMMARY_ARGS[@]}"

echo
echo "[DONE] Inviami soprattutto:"
echo "       $MAP_REPORT"
echo "       $OUTDIR/aggforce_training_summary.json"
echo "       $OUTDIR/aggforce_training_summary.csv"
