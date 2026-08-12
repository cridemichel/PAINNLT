#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${DNA_SELF_AUDIT_DIR:-dna_self_vs_intercopy}"
TARGET_FRAMES="${DNA_SELF_TARGET_FRAMES:-101}"
GAP_FRAMES="${DNA_SELF_GAP_FRAMES:-20}"
REUSE_DIR="${DNA_SELF_REUSE_DNA_ALL_DIR:-dna_only_conditional_noise}"

require_file(){ [ -f "$1" ] || { echo "[ERROR] File richiesto non trovato: $1" >&2; exit 1; }; }
for f in md.tpr md.gro md.trr tel22_dataset.bin tel22_training_config.json analyze_force_source_decomposition.py analyze_conditional_noise.py analyze_dna_self_vs_intercopy.py; do require_file "$f"; done
command -v gmx >/dev/null 2>&1 || { echo "[ERROR] gmx non trovato nel PATH." >&2; exit 1; }
mkdir -p "$OUTDIR"

echo "======================================================"
echo " TEL22: DNA SELF vs INTER-COPY CONDITIONAL-NOISE TEST"
echo "======================================================"
echo "[INFO] Output: $OUTDIR"
echo "[INFO] Target frames: $TARGET_FRAMES"
echo "[INFO] Same-copy minimum gap: $GAP_FRAMES raw frames"

"$PYTHON_BIN" analyze_dna_self_vs_intercopy.py --make-index --topology md.gro \
  --index-output "$OUTDIR/copy_seed.ndx" --index-manifest "$OUTDIR/copy_groups.json" \
  > "$OUTDIR/index_builder.log"
printf 'q\n' | gmx make_ndx -f md.tpr -n "$OUTDIR/copy_seed.ndx" -o "$OUTDIR/copy_groups.ndx" \
  > "$OUTDIR/make_ndx.log" 2>&1

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

# Reuse the already validated 03k DNA-all rerun when it has the same sampling.
DNA_ALL_GRO=""
DNA_ALL_TRR=""
if [ -f "$REUSE_DIR/dna_only.gro" ] && [ -f "$REUSE_DIR/dna_only_rerun.trr" ] && [ -f "$REUSE_DIR/subset_skip.txt" ]; then
  REUSE_SKIP=$(tr -d '[:space:]' < "$REUSE_DIR/subset_skip.txt")
  if [ "$REUSE_SKIP" = "$SKIP" ]; then
    DNA_ALL_GRO="$REUSE_DIR/dna_only.gro"
    DNA_ALL_TRR="$REUSE_DIR/dna_only_rerun.trr"
    echo "[INFO] Reusing DNA-all rerun from $REUSE_DIR (skip=$SKIP)"
  fi
fi

if [ -z "$DNA_ALL_TRR" ]; then
  G_DNA=$(awk '$1 ~ /^[0-9]+$/ && $2 == "FS_DNA_ALL" {print $1; exit}' "$OUTDIR/make_ndx.log")
  [ -n "$G_DNA" ] || { echo "[ERROR] FS_DNA_ALL group non trovato" >&2; exit 1; }
  echo "[INFO] Preparing DNA-all rerun (group $G_DNA)"
  printf '%s\n' "$G_DNA" | gmx convert-tpr -s md.tpr -n "$OUTDIR/copy_groups.ndx" -o "$OUTDIR/dna_all.tpr" \
    > "$OUTDIR/convert_tpr_dna_all.log" 2>&1
  printf '%s\n' "$G_DNA" | gmx trjconv -s md.tpr -f md.gro -n "$OUTDIR/copy_groups.ndx" -o "$OUTDIR/dna_all.gro" \
    > "$OUTDIR/trjconv_dna_all_topology.log" 2>&1
  printf '%s\n' "$G_DNA" | gmx trjconv -s md.tpr -f md.trr -n "$OUTDIR/copy_groups.ndx" \
    -o "$OUTDIR/dna_all_input.trr" -skip "$SKIP" -force > "$OUTDIR/trjconv_dna_all_input.log" 2>&1
  echo "[INFO] mdrun -rerun DNA-all"
  (
    cd "$OUTDIR"
    gmx mdrun -s dna_all.tpr -rerun dna_all_input.trr -deffnm dna_all_rerun > mdrun_dna_all.log 2>&1
  )
  DNA_ALL_GRO="$OUTDIR/dna_all.gro"
  DNA_ALL_TRR="$OUTDIR/dna_all_rerun.trr"
fi
require_file "$DNA_ALL_GRO"
require_file "$DNA_ALL_TRR"

NCOPIES=$($PYTHON_BIN - "$OUTDIR/copy_groups.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['copies'])
PY
)

for ((ci=0; ci<NCOPIES; ci++)); do
  tag=$(printf 'copy_%02d' "$ci")
  gname=$(printf 'FS_COPY_%02d' "$ci")
  gid=$(awk -v name="$gname" '$1 ~ /^[0-9]+$/ && $2 == name {print $1; exit}' "$OUTDIR/make_ndx.log")
  [ -n "$gid" ] || { echo "[ERROR] Gruppo $gname non trovato" >&2; exit 1; }
  echo "[INFO] Preparing self rerun: $tag (group $gid)"
  printf '%s\n' "$gid" | gmx convert-tpr -s md.tpr -n "$OUTDIR/copy_groups.ndx" -o "$OUTDIR/${tag}.tpr" \
    > "$OUTDIR/convert_tpr_${tag}.log" 2>&1
  printf '%s\n' "$gid" | gmx trjconv -s md.tpr -f md.gro -n "$OUTDIR/copy_groups.ndx" -o "$OUTDIR/${tag}.gro" \
    > "$OUTDIR/trjconv_${tag}_topology.log" 2>&1
  printf '%s\n' "$gid" | gmx trjconv -s md.tpr -f md.trr -n "$OUTDIR/copy_groups.ndx" \
    -o "$OUTDIR/${tag}_input.trr" -skip "$SKIP" -force > "$OUTDIR/trjconv_${tag}_input.log" 2>&1
  echo "[INFO] mdrun -rerun self: $tag"
  (
    cd "$OUTDIR"
    gmx mdrun -s "${tag}.tpr" -rerun "${tag}_input.trr" -deffnm "${tag}_rerun" > "mdrun_${tag}.log" 2>&1
  )
  require_file "$OUTDIR/${tag}_rerun.trr"
done

"$PYTHON_BIN" analyze_dna_self_vs_intercopy.py \
  --dataset tel22_dataset.bin --config tel22_training_config.json --raw-topology md.gro --raw-trr md.trr \
  --dna-all-topology "$DNA_ALL_GRO" --dna-all-rerun-trr "$DNA_ALL_TRR" \
  --copy-dir "$OUTDIR" --copy-manifest "$OUTDIR/copy_groups.json" \
  --same-copy-gap-frames "$GAP_FRAMES" \
  --output-json "$OUTDIR/dna_self_vs_intercopy_report.json" \
  --output-csv "$OUTDIR/dna_self_vs_intercopy_pairs.csv" | tee "$OUTDIR/analysis_stdout.txt"

echo "[DONE] $OUTDIR/dna_self_vs_intercopy_report.json"
