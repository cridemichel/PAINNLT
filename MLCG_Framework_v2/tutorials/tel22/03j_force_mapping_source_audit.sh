#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
OUTDIR="${FORCE_SOURCE_AUDIT_DIR:-force_source_audit}"
TARGET_FRAMES="${FORCE_SOURCE_TARGET_FRAMES:-11}"
CLOSURE_TOL="${FORCE_SOURCE_CLOSURE_TOL:-1e-3}"

require_file() {
    if [ ! -f "$1" ]; then
        echo "[ERROR] File richiesto non trovato: $1" >&2
        exit 1
    fi
}

for f in md.tpr md.gro md.trr tel22_dataset.bin; do require_file "$f"; done
if ! command -v gmx >/dev/null 2>&1; then
    echo "[ERROR] gmx non trovato nel PATH." >&2
    exit 1
fi

mkdir -p "$OUTDIR"

echo "======================================================"
echo " TEL22: AA->CG FORCE MAPPING + SOURCE DECOMPOSITION"
echo "======================================================"
echo "[INFO] Output: $OUTDIR"
echo "[INFO] Target subset frames: $TARGET_FRAMES"
echo "[INFO] Closure tolerance: $CLOSURE_TOL"

# Build exhaustive custom groups from md.gro. MDAnalysis may lag behind the
# newest GROMACS TPR/TPX format, while GRO is stable and is also the topology
# source used by this framework's atomistic->CG preprocessing.
"$PYTHON_BIN" analyze_force_source_decomposition.py \
    --make-index \
    --topology md.gro \
    --index-output "$OUTDIR/force_source_seed.ndx" \
    --index-manifest "$OUTDIR/force_source_groups.json" \
    > "$OUTDIR/index_builder.log"

# Merge custom groups with GROMACS defaults and capture deterministic group IDs.
printf 'q\n' | gmx make_ndx \
    -f md.tpr \
    -n "$OUTDIR/force_source_seed.ndx" \
    -o "$OUTDIR/force_source.ndx" \
    > "$OUTDIR/make_ndx.log" 2>&1

group_id() {
    local name="$1"
    local id
    id=$(awk -v n="$name" '$1 ~ /^[0-9]+$/ && $2 == n {print $1; exit}' "$OUTDIR/make_ndx.log")
    if [ -z "$id" ]; then
        echo "[ERROR] Impossibile determinare il group id per $name" >&2
        cat "$OUTDIR/make_ndx.log" >&2
        exit 1
    fi
    printf '%s' "$id"
}

G_DNA=$(group_id FS_DNA_ONLY)
G_DW=$(group_id FS_DNA_WATER)
G_DK=$(group_id FS_DNA_K)
G_DC=$(group_id FS_DNA_CL)

echo "[INFO] GROMACS groups: DNA=$G_DNA DNA+water=$G_DW DNA+K=$G_DK DNA+Cl=$G_DC"

# Count raw frames with MDAnalysis so the requested subset size is approximately stable.
NFRAMES=$($PYTHON_BIN - <<'PY'
import MDAnalysis as mda
u=mda.Universe('md.gro','md.trr')
print(len(u.trajectory))
PY
)
if [ "$TARGET_FRAMES" -lt 2 ]; then TARGET_FRAMES=2; fi
SKIP=$(( (NFRAMES - 1) / (TARGET_FRAMES - 1) ))
if [ "$SKIP" -lt 1 ]; then SKIP=1; fi
printf '%s\n' "$SKIP" > "$OUTDIR/subset_skip.txt"
echo "[INFO] md.trr frames=$NFRAMES -> trjconv -skip $SKIP"

# Full-system subset. Force records are retained for an independent structural check,
# although mdrun -rerun itself consumes coordinates only.
printf '0\n' | gmx trjconv \
    -s md.tpr -f md.trr \
    -o "$OUTDIR/full_subset.trr" -skip "$SKIP" -force \
    > "$OUTDIR/trjconv_full.log" 2>&1

make_subset() {
    local tag="$1"
    local gid="$2"
    echo "[INFO] Preparing subset: $tag"
    printf '%s\n' "$gid" | gmx convert-tpr \
        -s md.tpr -n "$OUTDIR/force_source.ndx" \
        -o "$OUTDIR/${tag}.tpr" \
        > "$OUTDIR/convert_tpr_${tag}.log" 2>&1
    # Matching GRO topology for MDAnalysis. This avoids parsing the generated
    # TPR in Python and guarantees the same atom order as the selected TRR.
    printf '%s\n' "$gid" | gmx trjconv \
        -s md.tpr -f md.gro -n "$OUTDIR/force_source.ndx" \
        -o "$OUTDIR/${tag}.gro" \
        > "$OUTDIR/trjconv_${tag}_topology.log" 2>&1
    printf '%s\n' "$gid" | gmx trjconv \
        -s md.tpr -f "$OUTDIR/full_subset.trr" -n "$OUTDIR/force_source.ndx" \
        -o "$OUTDIR/${tag}_input.trr" -force \
        > "$OUTDIR/trjconv_${tag}.log" 2>&1
}

make_subset dna_only "$G_DNA"
make_subset dna_water "$G_DW"
make_subset dna_k "$G_DK"
make_subset dna_cl "$G_DC"

run_rerun() {
    local tag="$1"
    local tpr="$2"
    local trr="$3"
    echo "[INFO] mdrun -rerun: $tag"
    (
        cd "$OUTDIR"
        gmx mdrun -s "$(basename "$tpr")" -rerun "$(basename "$trr")" \
            -deffnm "${tag}_rerun" \
            > "mdrun_${tag}.log" 2>&1
    )
}

# Full rerun establishes the reference on the exact same subset; four subset reruns
# permit source separation. These are single-point evaluations, not MD trajectories.
cp md.tpr "$OUTDIR/full.tpr"
cp md.gro "$OUTDIR/full.gro"
run_rerun full "$OUTDIR/full.tpr" "$OUTDIR/full_subset.trr"
run_rerun dna_only "$OUTDIR/dna_only.tpr" "$OUTDIR/dna_only_input.trr"
run_rerun dna_water "$OUTDIR/dna_water.tpr" "$OUTDIR/dna_water_input.trr"
run_rerun dna_k "$OUTDIR/dna_k.tpr" "$OUTDIR/dna_k_input.trr"
run_rerun dna_cl "$OUTDIR/dna_cl.tpr" "$OUTDIR/dna_cl_input.trr"

for trr in full_rerun.trr dna_only_rerun.trr dna_water_rerun.trr dna_k_rerun.trr dna_cl_rerun.trr; do
    require_file "$OUTDIR/$trr"
done

"$PYTHON_BIN" analyze_force_source_decomposition.py \
    --analyze \
    --full-topology "$OUTDIR/full.gro" \
    --raw-trr md.trr \
    --full-trr "$OUTDIR/full_rerun.trr" \
    --dna-topology "$OUTDIR/dna_only.gro" \
    --dna-trr "$OUTDIR/dna_only_rerun.trr" \
    --dna-water-topology "$OUTDIR/dna_water.gro" \
    --dna-water-trr "$OUTDIR/dna_water_rerun.trr" \
    --dna-k-topology "$OUTDIR/dna_k.gro" \
    --dna-k-trr "$OUTDIR/dna_k_rerun.trr" \
    --dna-cl-topology "$OUTDIR/dna_cl.gro" \
    --dna-cl-trr "$OUTDIR/dna_cl_rerun.trr" \
    --dataset tel22_dataset.bin \
    --closure-tol "$CLOSURE_TOL" \
    --output "$OUTDIR/force_mapping_source_report.json" \
    | tee "$OUTDIR/analysis_stdout.txt"

cat > "$OUTDIR/README.txt" <<'EOF'
TEL22 force-mapping / atomistic force-source audit
==================================================

Main result:
  force_mapping_source_report.json

Interpretation rule:
  Read source magnitudes ONLY if closure.status == PASS.
  The source decomposition is produced by exact GROMACS subset reruns:
    DNA-only          -> DNA-DNA contribution
    DNA+water - DNA   -> water-on-DNA contribution
    DNA+K - DNA       -> K-on-DNA contribution
    DNA+Cl - DNA      -> Cl-on-DNA contribution

The script also checks that the current per-residue COM coordinate mapping and
simple force sum satisfy B C^T = I for translations.

IMPORTANT: residual_minus_environment is a diagnostic quantity only. It is not
recommended as a new training target without a separate thermodynamic argument.
EOF

echo "[DONE] Report principale: $OUTDIR/force_mapping_source_report.json"
