#!/bin/bash
set -euo pipefail

# TEL22 Morse-prior learnability ablation.
#
# A_Morse_ON  : current production residual target
#               F_ref - F_harmonic - F_WCA - F_Morse
# B_Morse_OFF : exact same harmonic/WCA priors, but Morse entries removed
#               F_ref - F_harmonic - F_WCA
#
# The ON dataset/priors are reused from the current, already-regenerated TEL22
# artifacts.  The OFF dataset is generated with --priors from an exact copy of
# cg_priors.json with only type="morse" bonds removed.  This isolates Morse
# without refitting WCA/harmonic parameters.
#
# Environment overrides:
#   MORSE_ABLATION_EPOCHS=8        epochs per training case (default 8)
#   MORSE_ABLATION_SKIP_TRAIN=1    build/compare targets only
#   PYTHON_BIN=/path/to/python     Python with MDAnalysis/Numpy/SciPy

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILDER="$ROOT_DIR/preprocessing/build_cg_dataset.py"
TRAINER="$ROOT_DIR/training/build/train_painn"
PYTHON_BIN="${PYTHON_BIN:-/Users/demichel/PYTHON/bin/python}"
EPOCHS="${MORSE_ABLATION_EPOCHS:-8}"
SKIP_TRAIN="${MORSE_ABLATION_SKIP_TRAIN:-0}"
RUN_ROOT="$SCRIPT_DIR/morse_prior_ablation_runs"
ON_DIR="$RUN_ROOT/A_Morse_ON"
OFF_DIR="$RUN_ROOT/B_Morse_OFF"
BASE_CONFIG="$SCRIPT_DIR/tel22_training_config.json"
ON_DATASET="$SCRIPT_DIR/tel22_dataset.bin"
ON_PRIORS="$SCRIPT_DIR/cg_priors.json"

case "$EPOCHS" in
    ''|*[!0-9]*) echo "Errore: MORSE_ABLATION_EPOCHS deve essere un intero positivo"; exit 2 ;;
esac
if [ "$EPOCHS" -le 0 ]; then
    echo "Errore: MORSE_ABLATION_EPOCHS deve essere > 0"
    exit 2
fi
if [ "$SKIP_TRAIN" != "0" ] && [ "$SKIP_TRAIN" != "1" ]; then
    echo "Errore: MORSE_ABLATION_SKIP_TRAIN deve essere 0 o 1"
    exit 2
fi

for f in "$BUILDER" "$SCRIPT_DIR/md.gro" "$SCRIPT_DIR/md_whole.trr" \
         "$SCRIPT_DIR/tel22_topology.json" "$BASE_CONFIG" "$ON_DATASET" "$ON_PRIORS"; do
    if [ ! -f "$f" ]; then
        echo "Errore: file richiesto non trovato: $f"
        exit 1
    fi
done
if [ "$SKIP_TRAIN" = "0" ] && [ ! -x "$TRAINER" ]; then
    echo "Errore: trainer non trovato/eseguibile: $TRAINER"
    echo "Ricompila prima training/build/train_painn."
    exit 1
fi

mkdir -p "$RUN_ROOT"
rm -rf "$ON_DIR" "$OFF_DIR"
mkdir -p "$ON_DIR" "$OFF_DIR"

# Validate the current production priors and create a filtered copy.  We require
# the corrected explicit WCA topology so this ablation cannot accidentally
# resurrect the old Morse-as-1-2 exclusion bug.
"$PYTHON_BIN" - "$ON_PRIORS" "$OFF_DIR/priors_no_morse.json" <<'PY'
import json
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:3])
p = json.loads(src.read_text())
bonds = p.get("bonds", [])
morse = [b for b in bonds if str(b.get("type", "")).lower() == "morse"]
if not morse:
    raise SystemExit("[ERROR] Nessun Morse trovato in cg_priors.json: ablation non definita.")
meta = p.get("wca_exclusions", {})
if meta.get("pair_source") != "explicit_topology_pairs_v2":
    raise SystemExit("[ERROR] cg_priors.json non usa explicit_topology_pairs_v2.")
if int(meta.get("direct_pair_count", -1)) != 210 or int(meta.get("one_three_pair_count", -1)) != 200:
    raise SystemExit(
        "[ERROR] Attese exclusions TEL22 corrette 210/200; trovato "
        f"{meta.get('direct_pair_count')}/{meta.get('one_three_pair_count')}."
    )

direct = {tuple(sorted(map(int, x))) for x in meta.get("direct_pairs", [])}
morse_pairs = {
    tuple(sorted((int(b["mol_i"]), int(b["mol_j"]))))
    for b in morse
}
if direct & morse_pairs:
    raise SystemExit("[ERROR] Morse ancora presenti nelle WCA direct exclusions.")

q = json.loads(json.dumps(p))
q["bonds"] = [b for b in bonds if str(b.get("type", "")).lower() != "morse"]
q["ablation"] = {
    "name": "Morse_OFF",
    "source_priors": str(src),
    "removed_morse_bonds": len(morse),
    "all_other_priors_unchanged": True,
}
dst.write_text(json.dumps(q, indent=2) + "\n")
print(f"[INFO] Morse priors trovati/rimossi per OFF: {len(morse)}")
print(f"[INFO] Harmonic/WCA/angle/dihedral priors conservati identici: {dst}")
PY

# Keep immutable provenance copies for the ON case; no need to duplicate the
# large production dataset because training can read it directly.
cp "$ON_PRIORS" "$ON_DIR/cg_priors.json"
ln -s "$ON_DATASET" "$ON_DIR/tel22_dataset.bin"

# Build ONLY the OFF target using the exact ON priors with Morse removed.
# Run in OFF_DIR so generated rigid_bodies_info.json/cg artifacts cannot touch
# production files in tutorials/tel22.
echo "======================================================"
echo " 03h. MORSE PRIOR ABLATION: BUILD OFF DATASET"
echo "======================================================"
(
    cd "$OFF_DIR"
    "$PYTHON_BIN" "$BUILDER" \
        --topology "$SCRIPT_DIR/md.gro" \
        --trajectory "$SCRIPT_DIR/md_whole.trr" \
        --config "$SCRIPT_DIR/tel22_topology.json" \
        --priors "$OFF_DIR/priors_no_morse.json" \
        --output "$OFF_DIR/tel22_dataset.bin" \
        | tee build_off.log
)
cp "$OFF_DIR/priors_no_morse.json" "$OFF_DIR/cg_priors.json"

# Structural + numerical proof that the two datasets differ only in targets.
"$PYTHON_BIN" "$SCRIPT_DIR/analyze_morse_ablation.py" \
    --on "$ON_DATASET" \
    --off "$OFF_DIR/tel22_dataset.bin" \
    --output-json "$RUN_ROOT/morse_target_ablation.json" \
    --output-csv "$RUN_ROOT/morse_target_ablation_frames.csv"

if [ "$SKIP_TRAIN" = "1" ]; then
    echo
    echo "[INFO] MORSE_ABLATION_SKIP_TRAIN=1: target diagnostics completati, training saltato."
    exit 0
fi

make_config() {
    local dst="$1"
    "$PYTHON_BIN" - "$BASE_CONFIG" "$dst" "$EPOCHS" <<'PY'
import json
import sys
src, dst, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as fh:
    cfg = json.load(fh)
# Fix all optimization/model knobs identically in both cases.  batch=4 and
# periodic MPS cache cleanup are intentionally retained for the 48-GB Mac.
cfg.update({
    "epochs": epochs,
    "batch_size": 4,
    "torque_weight": 0.5,
    "grad_clip_norm": 1.0,
    "diagnostic_overfit_frames": 0,
    "physical_validation_only": True,
    "include_decoys_in_train": False,
    "shuffle_each_epoch": True,
    "report_grad_norms": True,
    "mps_empty_cache_every_batches": 4,
    "split_seed": 42,
    "validation_fraction": 0.2,
    # Keep the short comparison free from scheduler/early-stop differences.
    "early_stopping_patience": max(50, epochs + 5),
    "reduce_lr_patience": max(50, epochs + 5),
})
with open(dst, "w") as fh:
    json.dump(cfg, fh, indent=4)
    fh.write("\n")
PY
}

run_case() {
    local label="$1"
    local case_dir="$2"
    local dataset="$3"
    make_config "$case_dir/config.json"
    rm -f "$case_dir/model.pt" "$case_dir/model.pt.manifest.json" "$case_dir/cg_training_log.csv"
    echo
    echo "------------------------------------------------------"
    echo " $label | epochs=$EPOCHS | batch=4 | torque_weight=0.5"
    echo "------------------------------------------------------"
    (
        cd "$case_dir"
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        "$TRAINER" "$dataset" model.pt config.json | tee run.log
    )
    if ! grep -q "Detected physical frames: 1001" "$case_dir/run.log"; then
        echo "[ERROR] $label: attesi 1001 frame fisici."
        exit 3
    fi
    if ! grep -q "Detected zero-target OOD decoys: 0" "$case_dir/run.log"; then
        echo "[ERROR] $label: attesi 0 decoy."
        exit 3
    fi
}

run_case "A_Morse_ON" "$ON_DIR" "$ON_DATASET"
run_case "B_Morse_OFF" "$OFF_DIR" "$OFF_DIR/tel22_dataset.bin"

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_morse_ablation.py" \
    --run-root "$RUN_ROOT" \
    --output "$SCRIPT_DIR/morse_prior_ablation_summary.csv"

echo
echo "======================================================"
echo " MORSE PRIOR ABLATION COMPLETATA"
echo "======================================================"
echo "Target report:  $RUN_ROOT/morse_target_ablation.json"
echo "Training CSV:   $SCRIPT_DIR/morse_prior_ablation_summary.csv"
echo
echo "Interpretazione primaria: confronta improvement_F_vs_zero e improvement_T_vs_zero."
echo "Se OFF >> ON, il Morse rende il residuale meno apprendibile; se ON ~ OFF, non e la causa."
