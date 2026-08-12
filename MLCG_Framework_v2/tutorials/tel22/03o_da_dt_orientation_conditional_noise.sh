#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_ORIENT_DIR:-dna_da_dt_orientation_noise}"
COPY_DIR="${DNA_SELF_FULL_RERUN_DIR:-dna_self_full_reruns}"
SCALES="${DNA_ORIENT_SCALES_NM:-0.10 0.20 0.30}"
GAP="${DNA_ORIENT_SAME_COPY_GAP_FRAMES:-20}"
SEED="${DNA_ORIENT_SEED:-20260812}"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in tel22_dataset.bin md.gro md.trr \
         analyze_conditional_noise.py analyze_dna_self_vs_intercopy.py analyze_force_source_decomposition.py \
         analyze_da_dt_orientation_noise.py \
         "$COPY_DIR/copy_groups.json" "$COPY_DIR/full_self_rerun_report.json"; do
    require_file "$f"
done

case "$GAP" in ''|*[!0-9]*) echo "[ERROR] DNA_ORIENT_SAME_COPY_GAP_FRAMES deve essere intero >=1" >&2; exit 2;; esac
[ "$GAP" -ge 1 ] || { echo "[ERROR] gap deve essere >=1" >&2; exit 2; }
case "$SEED" in ''|*[!0-9]*) echo "[ERROR] DNA_ORIENT_SEED deve essere intero" >&2; exit 2;; esac

NCOPIES=$($PYTHON_BIN - "$COPY_DIR/copy_groups.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['copies']))
PY
)
for ((ci=0; ci<NCOPIES; ci++)); do
    tag=$(printf 'copy_%02d' "$ci")
    require_file "$COPY_DIR/${tag}.gro"
    require_file "$COPY_DIR/${tag}_rerun.trr"
done

mkdir -p "$OUTDIR"
read -r -a SCALE_ARGS <<< "$SCALES"

printf '%s\n' \
  "======================================================" \
  " 03o. TEL22 DA/DT ORIENTATION CONDITIONAL NOISE" \
  "======================================================" \
  "No new GROMACS rerun and no neural-network training." \
  "Self targets: $COPY_DIR" \
  "Orientation scales [nm]: $SCALES" \
  "Same-copy minimum frame gap: $GAP" \
  "" \
  "The test compares the current CG descriptor against CG + rigid DA/DT base orientation." \
  "DA anchors: N9,C4,N1 | DT anchors: N1,C4,C6" \
  "Only orientation is added: anchor distances are normalized away." \
  ""

"$PYTHON_BIN" analyze_da_dt_orientation_noise.py \
  --dataset tel22_dataset.bin \
  --raw-topology md.gro \
  --raw-trr md.trr \
  --copy-dir "$COPY_DIR" \
  --copy-manifest "$COPY_DIR/copy_groups.json" \
  --same-copy-gap-frames "$GAP" \
  --orientation-scales-nm "${SCALE_ARGS[@]}" \
  --seed "$SEED" \
  --output-json "$OUTDIR/da_dt_orientation_noise_report.json" \
  --output-csv "$OUTDIR/da_dt_orientation_noise_pairs.csv"

echo
echo "[DONE] Inviami soprattutto:"
echo "       $OUTDIR/da_dt_orientation_noise_report.json"
echo "       $OUTDIR/da_dt_orientation_noise_pairs.csv"
