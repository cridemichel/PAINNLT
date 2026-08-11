#!/bin/bash
set -euo pipefail

# Offline TEL22 conditional-noise / local-neighbor diagnostic.
# No GROMACS rerun, no dataset rewrite, no PaiNN training.
#
# Optional overrides:
#   PYTHON_BIN=/path/to/python
#   CONDITIONAL_NOISE_GAP_FRAMES=20
#   CONDITIONAL_NOISE_CUTOFF=1.2616   # normally read from training config

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/demichel/PYTHON/bin/python}"
DATASET="$SCRIPT_DIR/tel22_dataset.bin"
CONFIG="$SCRIPT_DIR/tel22_training_config.json"
OUT_DIR="$SCRIPT_DIR/conditional_noise_runs"
GAP="${CONDITIONAL_NOISE_GAP_FRAMES:-20}"
CUTOFF="${CONDITIONAL_NOISE_CUTOFF:-}"

for f in "$DATASET" "$CONFIG" "$SCRIPT_DIR/analyze_conditional_noise.py"; do
    if [ ! -f "$f" ]; then
        echo "Errore: file richiesto non trovato: $f"
        exit 1
    fi
done
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Errore: Python non trovato/eseguibile: $PYTHON_BIN"
    exit 1
fi
case "$GAP" in
    ''|*[!0-9]*) echo "Errore: CONDITIONAL_NOISE_GAP_FRAMES deve essere un intero positivo"; exit 2 ;;
esac
if [ "$GAP" -le 0 ]; then
    echo "Errore: CONDITIONAL_NOISE_GAP_FRAMES deve essere > 0"
    exit 2
fi

mkdir -p "$OUT_DIR"

ARGS=(
    --dataset "$DATASET"
    --config "$CONFIG"
    --same-copy-gap-frames "$GAP"
    --output-json "$OUT_DIR/conditional_noise_report.json"
    --output-csv "$OUT_DIR/conditional_noise_pairs.csv"
)
if [ -n "$CUTOFF" ]; then
    ARGS+=(--cutoff "$CUTOFF")
fi

echo "======================================================"
echo " 03i. TEL22 CONDITIONAL-NOISE DIAGNOSTIC"
echo "======================================================"
echo "Dataset: $(basename "$DATASET")"
echo "Training: nessuno"
echo "GROMACS:  nessun rerun"
echo "Same-copy minimum time gap: $GAP frame"
echo

"$PYTHON_BIN" "$SCRIPT_DIR/analyze_conditional_noise.py" "${ARGS[@]}"

echo
echo "Completato. Invia questi due file:"
echo "  $OUT_DIR/conditional_noise_report.json"
echo "  $OUT_DIR/conditional_noise_pairs.csv"
