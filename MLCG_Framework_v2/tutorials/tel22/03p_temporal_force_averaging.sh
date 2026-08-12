#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_TEMPORAL_AVG_DIR:-dna_temporal_force_averaging}"
COPY_DIR="${DNA_SELF_FULL_RERUN_DIR:-dna_self_full_reruns}"
WINDOWS="${DNA_TEMPORAL_AVG_WINDOWS_PS:-1 2 5 10}"
GAP="${DNA_TEMPORAL_AVG_SAME_COPY_GAP_FRAMES:-20}"
SEED="${DNA_TEMPORAL_AVG_SEED:-20260812}"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in tel22_dataset.bin md.gro md.trr \
         analyze_conditional_noise.py analyze_dna_self_vs_intercopy.py analyze_force_source_decomposition.py \
         analyze_temporal_force_averaging.py \
         "$COPY_DIR/copy_groups.json" "$COPY_DIR/full_self_rerun_report.json"; do
    require_file "$f"
done

case "$GAP" in ''|*[!0-9]*) echo "[ERROR] DNA_TEMPORAL_AVG_SAME_COPY_GAP_FRAMES deve essere intero >=1" >&2; exit 2;; esac
[ "$GAP" -ge 1 ] || { echo "[ERROR] gap deve essere >=1" >&2; exit 2; }
case "$SEED" in ''|*[!0-9]*) echo "[ERROR] DNA_TEMPORAL_AVG_SEED deve essere intero" >&2; exit 2;; esac

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
read -r -a WINDOW_ARGS <<< "$WINDOWS"

printf '%s\n' \
  "======================================================" \
  " 03p. TEL22 TEMPORAL SELF-FORCE AVERAGING" \
  "======================================================" \
  "No new GROMACS rerun and no neural-network training." \
  "Self targets: $COPY_DIR" \
  "Centered boxcar widths [ps]: $WINDOWS" \
  "Same-copy minimum frame gap: $GAP" \
  "" \
  "All windows use the same central frames and the same nearest/random pairs." \
  "Each support-frame force is first Kabsch-transported to the common copy frame." \
  "For 1-ps input sampling, the 1-ps boxcar is exactly the instantaneous baseline." \
  ""

"$PYTHON_BIN" analyze_temporal_force_averaging.py \
  --dataset tel22_dataset.bin \
  --raw-topology md.gro \
  --raw-trr md.trr \
  --copy-dir "$COPY_DIR" \
  --copy-manifest "$COPY_DIR/copy_groups.json" \
  --window-ps "${WINDOW_ARGS[@]}" \
  --same-copy-gap-frames "$GAP" \
  --seed "$SEED" \
  --output-json "$OUTDIR/temporal_force_averaging_report.json" \
  --output-csv "$OUTDIR/temporal_force_averaging_pairs.csv"

echo
echo "[DONE] Inviami soprattutto:"
echo "       $OUTDIR/temporal_force_averaging_report.json"
echo "       $OUTDIR/temporal_force_averaging_pairs.csv"
