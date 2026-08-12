#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_NOISE_AUDIT_DIR:-dna_only_conditional_noise}"
TARGET_FRAMES="${DNA_NOISE_TARGET_FRAMES:-101}"
GAP_FRAMES="${DNA_NOISE_GAP_FRAMES:-20}"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in md.tpr md.gro md.trr tel22_dataset.bin tel22_training_config.json analyze_force_source_decomposition.py analyze_conditional_noise.py analyze_dna_only_conditional_noise.py; do require_file "$f"; done
command -v gmx >/dev/null 2>&1 || { echo "[ERROR] gmx non trovato nel PATH." >&2; exit 1; }
mkdir -p "$OUTDIR"

echo "======================================================"
echo " TEL22: DNA-ONLY CONDITIONAL-NOISE TEST"
echo "======================================================"
echo "[INFO] Output: $OUTDIR"
echo "[INFO] Target frames: $TARGET_FRAMES"
echo "[INFO] Same-copy minimum gap: $GAP_FRAMES raw frames"

"$PYTHON_BIN" analyze_force_source_decomposition.py --make-index --topology md.gro \
  --index-output "$OUTDIR/force_source_seed.ndx" --index-manifest "$OUTDIR/force_source_groups.json" \
  > "$OUTDIR/index_builder.log"
printf 'q\n' | gmx make_ndx -f md.tpr -n "$OUTDIR/force_source_seed.ndx" -o "$OUTDIR/force_source.ndx" \
  > "$OUTDIR/make_ndx.log" 2>&1
G_DNA=$(awk '$1 ~ /^[0-9]+$/ && $2 == "FS_DNA_ONLY" {print $1; exit}' "$OUTDIR/make_ndx.log")
[ -n "$G_DNA" ] || { echo "[ERROR] FS_DNA_ONLY group non trovato" >&2; exit 1; }
echo "[INFO] GROMACS DNA group: $G_DNA"

NFRAMES=$($PYTHON_BIN - <<'PY'
import MDAnalysis as mda
u=mda.Universe('md.gro','md.trr')
print(len(u.trajectory))
PY
)
if [ "$TARGET_FRAMES" -lt 2 ]; then TARGET_FRAMES=2; fi
SKIP=$(( (NFRAMES - 1) / (TARGET_FRAMES - 1) )); [ "$SKIP" -ge 1 ] || SKIP=1
printf '%s\n' "$SKIP" > "$OUTDIR/subset_skip.txt"
echo "[INFO] md.trr frames=$NFRAMES -> trjconv -skip $SKIP"

printf '%s\n' "$G_DNA" | gmx convert-tpr -s md.tpr -n "$OUTDIR/force_source.ndx" -o "$OUTDIR/dna_only.tpr" \
  > "$OUTDIR/convert_tpr_dna_only.log" 2>&1
printf '%s\n' "$G_DNA" | gmx trjconv -s md.tpr -f md.gro -n "$OUTDIR/force_source.ndx" -o "$OUTDIR/dna_only.gro" \
  > "$OUTDIR/trjconv_dna_topology.log" 2>&1
printf '%s\n' "$G_DNA" | gmx trjconv -s md.tpr -f md.trr -n "$OUTDIR/force_source.ndx" \
  -o "$OUTDIR/dna_only_input.trr" -skip "$SKIP" -force > "$OUTDIR/trjconv_dna_input.log" 2>&1

echo "[INFO] mdrun -rerun DNA-only"
(
 cd "$OUTDIR"
 gmx mdrun -s dna_only.tpr -rerun dna_only_input.trr -deffnm dna_only_rerun > mdrun_dna_only.log 2>&1
)
require_file "$OUTDIR/dna_only_rerun.trr"

"$PYTHON_BIN" analyze_dna_only_conditional_noise.py \
  --dataset tel22_dataset.bin --config tel22_training_config.json --raw-topology md.gro --raw-trr md.trr \
  --dna-topology "$OUTDIR/dna_only.gro" --dna-rerun-trr "$OUTDIR/dna_only_rerun.trr" \
  --same-copy-gap-frames "$GAP_FRAMES" \
  --output-json "$OUTDIR/dna_only_conditional_noise_report.json" \
  --output-csv "$OUTDIR/dna_only_conditional_noise_pairs.csv" | tee "$OUTDIR/analysis_stdout.txt"

echo "[DONE] $OUTDIR/dna_only_conditional_noise_report.json"
