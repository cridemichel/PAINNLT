#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${HERE}/model_config.sh" || -f "${HERE}/cg_priors.json" ]]; then
  TUTORIAL_DIR="${HERE}"
else
  TUTORIAL_DIR="$(cd "${HERE}/../.." && pwd)"
fi
ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
source "${TUTORIAL_DIR}/model_config.sh"
load_model_dependent_config step38 step39
cd "${TUTORIAL_DIR}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"

MODE="${1:-}"
case "${MODE}" in
  --dry-run) PY_MODE="dry-run" ;;
  --run)     PY_MODE="run" ;;
  --resume)  PY_MODE="resume" ;;
  *)
    echo "Usage: $0 --dry-run|--run|--resume" >&2
    exit 2
    ;;
esac

STEP38_OUT="${IBI_DIHEDRAL_REPLICA_STEP38_OUT}"
IBI_REPORT="${STEP38_OUT}/ibi/ibi_report.json"
FINAL_SAMPLING_REPORT="${STEP38_OUT}/final_sampling_protocol.json"
FINAL_SAMPLE="${STEP38_OUT}/final_nvt/trajectory.npz"
TARGET_DATASET="${IBI_DIHEDRAL_REPLICA_TARGET_DATASET}"
CONFIG="${IBI_DIHEDRAL_REPLICA_CONFIG}"
RB_INFO="${IBI_DIHEDRAL_REPLICA_RB_INFO}"
SETTINGS="${IBI_DIHEDRAL_REPLICA_SETTINGS}"
OUT="${IBI_DIHEDRAL_REPLICA_OUT}"
REPORT="${OUT}/dihedral_ibi_replica_matrix_report.json"
CONFIG_PROVENANCE="${OUT}/model_config_provenance.json"
REPLICAS="${IBI_DIHEDRAL_REPLICA_COUNT}"
OVERWRITE="${OVERWRITE:-0}"

for path in "${IBI_REPORT}" "${FINAL_SAMPLING_REPORT}" "${FINAL_SAMPLE}" "${TARGET_DATASET}" "${CONFIG}" "${RB_INFO}" "${SETTINGS}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing required artifact: ${path}" >&2; exit 1; }
done
[[ -x "${PYPRESSO}" ]] || { echo "[ERROR] pypresso not executable: ${PYPRESSO}" >&2; exit 1; }

if [[ "${MODE}" == "--run" && -f "${REPORT}" && "${OVERWRITE}" != "1" ]]; then
  echo "[ERROR] Step-39 report already exists: ${REPORT}" >&2
  echo "        Use --resume, or OVERWRITE=1 --run to discard only step-39 generated runs." >&2
  exit 1
fi
if [[ "${MODE}" == "--run" && "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUT}"
fi

cat <<EOF
[STEP 39 -- MATCHED REPLICA TEST FOR CONSERVATIVE DIHEDRAL IBI]
step38 IBI report      : ${IBI_REPORT}
step38 final sample    : ${FINAL_SAMPLE}
target dataset         : ${TARGET_DATASET}
replica count          : ${REPLICAS} paired seed-pairs
reuse/new MD            : determined from exact prior/seed cells in the step-38 report
output                  : ${OUT}
[NOTE] Test only. No priors are modified or promoted; no NVE is run.
[NOTE] Same velocity/thermostat seed pairs are applied to every prior.
EOF

if [[ "${MODE}" != "--dry-run" ]]; then
  mkdir -p "${OUT}"
  write_model_dependent_provenance "${CONFIG_PROVENANCE}"
fi

"${PYTHON_BIN}" "${ROOT}/simulation/dihedral_ibi_replica_matrix.py" \
  --mode "${PY_MODE}" \
  --ibi-report "${IBI_REPORT}" \
  --final-sampling-report "${FINAL_SAMPLING_REPORT}" \
  --final-sample-npz "${FINAL_SAMPLE}" \
  --dataset "${TARGET_DATASET}" \
  --config "${CONFIG}" \
  --rb-info "${RB_INFO}" \
  --ibi-config "${SETTINGS}" \
  --pypresso "${PYPRESSO}" \
  --outdir "${OUT}" \
  --replicas "${REPLICAS}" \
  --model-config-provenance "${CONFIG_PROVENANCE}" \
  --output "${REPORT}"

if [[ "${MODE}" != "--dry-run" ]]; then
  echo "[DONE] step-39 matched replica report: ${REPORT}"
  echo "[NOTE] n=${REPLICAS} paired replicas is a configured diagnostic uncertainty estimate, not a universal promotion/certification gate."
fi
