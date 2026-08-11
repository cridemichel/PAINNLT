#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/Users/demichel/PYTHON/bin/python}"
AUDIT_DIR="${GMX_AUDIT_DIR:-gromacs_force_audit}"
TARGET_RERUN_FRAMES="${GMX_AUDIT_RERUN_FRAMES:-11}"
RERUN_REL_TOL="${GMX_AUDIT_RERUN_REL_TOL:-1e-3}"
RAW_WHOLE_REL_TOL="${GMX_AUDIT_RAW_WHOLE_REL_TOL:-1e-7}"
AUDITOR="${SCRIPT_DIR}/audit_gromacs_forces.py"

printf '%s\n' "======================================================"
printf '%s\n' " 01b. GROMACS FORCE/TRR AUDIT"
printf '%s\n' "======================================================"

if ! command -v gmx >/dev/null 2>&1; then
    echo "[ERRORE] gmx non trovato nel PATH."
    exit 1
fi
if [[ "${PYTHON_BIN}" == */* ]]; then
    if [ ! -x "${PYTHON_BIN}" ]; then
        echo "[ERRORE] Python non eseguibile: ${PYTHON_BIN}"
        exit 1
    fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERRORE] Python non trovato nel PATH: ${PYTHON_BIN}"
    exit 1
fi
if [ ! -f "${AUDITOR}" ]; then
    echo "[ERRORE] Auditor Python non trovato: ${AUDITOR}"
    exit 1
fi

for f in md.tpr md.trr md_whole.trr md.gro; do
    if [ ! -s "${f}" ]; then
        echo "[ERRORE] File richiesto mancante o vuoto: ${f}"
        exit 1
    fi
done

rm -rf "${AUDIT_DIR}"
mkdir -p "${AUDIT_DIR}"

GMX_VERSION="$(gmx --version 2>/dev/null | head -n 1 || true)"
echo "[INFO] ${GMX_VERSION:-GROMACS version unavailable}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Audit directory: ${AUDIT_DIR}"

# TPR is the source of truth for the parameters actually used by mdrun.
echo "[1/5] Dump dei parametri realmente contenuti in md.tpr..."
gmx dump -s md.tpr -om "${AUDIT_DIR}/md_from_tpr.mdp" > "${AUDIT_DIR}/md_tpr.dump" 2>&1

# Keep the native gmx check output for independent inspection.
echo "[2/5] gmx check sui TRR raw e whole..."
gmx check -f md.trr > "${AUDIT_DIR}/gmx_check_md_trr.txt" 2>&1
gmx check -f md_whole.trr > "${AUDIT_DIR}/gmx_check_md_whole_trr.txt" 2>&1

# Streaming MDAnalysis audit: positions+forces on every frame and raw-vs-whole forces.
echo "[3/5] Controllo frame-per-frame coordinate/forze e raw -> whole..."
set +e
"${PYTHON_BIN}" "${AUDITOR}" \
    --topology md.gro \
    --raw md.trr \
    --whole md_whole.trr \
    --mdp-from-tpr "${AUDIT_DIR}/md_from_tpr.mdp" \
    --output-dir "${AUDIT_DIR}" \
    --target-rerun-frames "${TARGET_RERUN_FRAMES}" \
    --raw-whole-rel-tol "${RAW_WHOLE_REL_TOL}" \
    --rerun-rel-tol "${RERUN_REL_TOL}" \
    --write-skip-file "${AUDIT_DIR}/rerun_skip.txt"
PRECHECK_STATUS=$?
set -e

if [ "${PRECHECK_STATUS}" -eq 2 ]; then
    echo "[ERRORE] Audit strutturale fallito; non eseguo il rerun."
    exit 2
fi

SKIP="$(tr -d '[:space:]' < "${AUDIT_DIR}/rerun_skip.txt")"
if ! [[ "${SKIP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERRORE] Valore -skip non valido prodotto dall'auditor: '${SKIP}'"
    exit 1
fi

echo "[INFO] Rerun subset: target=${TARGET_RERUN_FRAMES} frame, gmx trjconv -skip ${SKIP}."

# Preserve the original stored forces in the subset; mdrun -rerun uses the
# coordinates/box as input and recomputes forces from md.tpr.
echo "[4/5] Estrazione subset e ricalcolo indipendente con mdrun -rerun..."
printf '0\n' | gmx trjconv \
    -s md.tpr \
    -f md.trr \
    -o "${AUDIT_DIR}/stored_subset.trr" \
    -skip "${SKIP}" \
    -force \
    > "${AUDIT_DIR}/trjconv_subset.log" 2>&1

if [ ! -s "${AUDIT_DIR}/stored_subset.trr" ]; then
    echo "[ERRORE] Impossibile creare stored_subset.trr."
    exit 1
fi

(
    cd "${AUDIT_DIR}"
    # Avoid littering the directory with automatic #backup# files on reruns.
    export GMX_MAXBACKUP=-1
    gmx mdrun \
        -s ../md.tpr \
        -rerun stored_subset.trr \
        -deffnm rerun \
        > mdrun_rerun.log 2>&1
)

if [ ! -s "${AUDIT_DIR}/rerun.trr" ]; then
    echo "[ERRORE] mdrun -rerun non ha prodotto ${AUDIT_DIR}/rerun.trr."
    echo "         Controlla ${AUDIT_DIR}/mdrun_rerun.log"
    exit 1
fi

gmx check -f "${AUDIT_DIR}/stored_subset.trr" > "${AUDIT_DIR}/gmx_check_stored_subset.txt" 2>&1
gmx check -f "${AUDIT_DIR}/rerun.trr" > "${AUDIT_DIR}/gmx_check_rerun.txt" 2>&1

# Re-run the full audit including the independent force comparison.
echo "[5/5] Confronto forze archiviate vs forze ricalcolate..."
set +e
"${PYTHON_BIN}" "${AUDITOR}" \
    --topology md.gro \
    --raw md.trr \
    --whole md_whole.trr \
    --mdp-from-tpr "${AUDIT_DIR}/md_from_tpr.mdp" \
    --output-dir "${AUDIT_DIR}" \
    --target-rerun-frames "${TARGET_RERUN_FRAMES}" \
    --stored-subset "${AUDIT_DIR}/stored_subset.trr" \
    --rerun "${AUDIT_DIR}/rerun.trr" \
    --raw-whole-rel-tol "${RAW_WHOLE_REL_TOL}" \
    --rerun-rel-tol "${RERUN_REL_TOL}"
FINAL_STATUS=$?
set -e

cat > "${AUDIT_DIR}/README.txt" <<EOF
GROMACS force audit outputs
===========================

Primary report:
  gromacs_force_audit.json

Per-frame numerical diagnostics:
  raw_vs_whole_frames.csv
  stored_vs_rerun_frames.csv

Native GROMACS diagnostics:
  md_tpr.dump
  md_from_tpr.mdp
  gmx_check_md_trr.txt
  gmx_check_md_whole_trr.txt
  gmx_check_stored_subset.txt
  gmx_check_rerun.txt
  trjconv_subset.log
  mdrun_rerun.log

Interpretation:
- raw -> whole should preserve forces to numerical precision.
- stored -> rerun compares original TRR forces with forces recomputed from the
  exact saved coordinates using md.tpr. The default relative-RMS acceptance
  tolerance is ${RERUN_REL_TOL}; override with GMX_AUDIT_RERUN_REL_TOL.
- nstxout != nstfout is reported as WARN, but the decisive structural check is
  that every TRR frame actually consumed contains both positions and forces.
EOF

echo "======================================================"
echo " Audit completato."
echo " Report principale: ${AUDIT_DIR}/gromacs_force_audit.json"
echo " CSV rerun:          ${AUDIT_DIR}/stored_vs_rerun_frames.csv"
echo " Log rerun:          ${AUDIT_DIR}/mdrun_rerun.log"
echo "======================================================"

exit "${FINAL_STATUS}"
