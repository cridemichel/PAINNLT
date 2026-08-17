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
load_model_dependent_config step38
cd "${TUTORIAL_DIR}"
PYPRESSO="${PYPRESSO:-${ROOT}/espresso/build/pypresso}"

MODE="${1:-}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--run" && "${MODE}" != "--resume" ]]; then
  echo "Usage: $0 --dry-run|--run|--resume" >&2
  exit 2
fi

BASE_PRIORS="${IBI_DIHEDRAL_LOOP_BASE_PRIORS}"
TARGET_DATASET="${IBI_DIHEDRAL_LOOP_TARGET_DATASET}"
CONFIG="${IBI_DIHEDRAL_LOOP_CONFIG}"
RB_INFO="${IBI_DIHEDRAL_LOOP_RB_INFO}"
SETTINGS="${IBI_DIHEDRAL_LOOP_SETTINGS}"
STEP35_REPORT="${IBI_DIHEDRAL_LOOP_STEP35_REPORT}"
STEP37_REPORT="${IBI_DIHEDRAL_LOOP_STEP37_REPORT}"
OUT="${IBI_DIHEDRAL_LOOP_OUT}"
ITERATIONS="${IBI_DIHEDRAL_LOOP_ITERATIONS}"
OVERWRITE="${OVERWRITE:-0}"

if ! [[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] IBI_DIHEDRAL_LOOP_ITERATIONS must be a positive integer" >&2
  exit 2
fi

SEED_DIR="${OUT}/seed"
SEED_PRIORS="${SEED_DIR}/cg_priors_dihedral_ibi_seed.json"
SEED_REPORT="${SEED_DIR}/seed_report.json"
IBI_OUT="${OUT}/ibi"
IBI_REPORT="${IBI_OUT}/ibi_report.json"
FINAL_PRIORS="${IBI_OUT}/cg_priors_final.json"
PARITY_REPORT="${OUT}/runtime_parity_report.json"
NVT_DIR="${OUT}/final_nvt"
NVT_SAMPLE="${NVT_DIR}/trajectory.npz"
FINAL_SAMPLING_REPORT="${OUT}/final_sampling_protocol.json"
STRUCTURE_REPORT="${OUT}/runtime_structure_report.json"
FINAL_REPORT="${OUT}/conservative_in_loop_dihedral_ibi_test_report.json"
PRE_MATCHED_BACKUP="${OUT}/final_sampling_pre_matched_hotfix"
CONFIG_PROVENANCE="${OUT}/model_config_provenance.json"

for path in "${BASE_PRIORS}" "${TARGET_DATASET}" "${CONFIG}" "${RB_INFO}" "${SETTINGS}"; do
  [[ -f "${path}" ]] || { echo "[ERROR] Missing required artifact: ${path}" >&2; exit 1; }
done
[[ -x "${PYPRESSO}" ]] || { echo "[ERROR] pypresso not executable: ${PYPRESSO}" >&2; exit 1; }

"${PYPRESSO}" - <<'PY'
import espressomd
for name in ("ConservativeSplineDistance", "ConservativeSplineAngle", "ConservativeSplineDihedral"):
    if not hasattr(espressomd.interactions, name):
        raise RuntimeError(f"Missing espressomd.interactions.{name}")
print("[PASS] ESPResSo conservative distance/angle/dihedral bindings are available.")
PY

cat <<EOF
[STEP 38 -- CONSERVATIVE-IN-THE-LOOP PERIODIC DIHEDRAL IBI]
base production priors : ${BASE_PRIORS}
target geometry dataset: ${TARGET_DATASET}
IBI iterations          : ${ITERATIONS}
IBI settings            : ${SETTINGS}
runtime representation  : ConservativeSplineDihedral from iteration_000 onward
final candidate         : ${FINAL_PRIORS}
final matched sampling  : same burn-in/production protocol as the IBI loop
output                  : ${OUT}
[NOTE] Test only: production priors are never modified or promoted.
[NOTE] Active torsional IBI sampling is fail-closed against legacy TabulatedDihedral.
[NOTE] No NVE certification is performed here; first establish whether conservative-in-loop IBI improves torsional structure.
EOF

prepare_seed() {
  mkdir -p "${SEED_DIR}"
  "${PYTHON_BIN}" "${ROOT}/ibi/prepare_dihedral_ibi_test_seed.py" \
    --base-priors "${BASE_PRIORS}" --output "${SEED_PRIORS}" --report "${SEED_REPORT}" --grouping-strategy "${IBI_DIHEDRAL_LOOP_GROUPING_STRATEGY}"
}

run_ibi_loop() {
  "${PYTHON_BIN}" "${ROOT}/ibi/run_ibi_loop.py" \
    --dataset "${TARGET_DATASET}" --priors "${SEED_PRIORS}" \
    --config "${CONFIG}" --rb_info "${RB_INFO}" --iterations "${ITERATIONS}" \
    --outdir "${IBI_OUT}" --ibi-config "${SETTINGS}" --pypresso "${PYPRESSO}" \
    --neighbor_search "${IBI_DIHEDRAL_LOOP_NEIGHBOR_SEARCH}" --velocity_seed "${IBI_DIHEDRAL_LOOP_VELOCITY_SEED}" --thermostat_seed "${IBI_DIHEDRAL_LOOP_THERMOSTAT_SEED}" \
    --conservative-dihedrals-in-loop
}

archive_pre_matched_final_sampling() {
  # The first resume after this hotfix may contain the old 1000-step/checkpoint
  # final sample. Preserve it once for provenance before replacing it.
  if [[ ! -f "${FINAL_SAMPLING_REPORT}" && ! -e "${PRE_MATCHED_BACKUP}" ]]; then
    if [[ -d "${NVT_DIR}" || -f "${STRUCTURE_REPORT}" || -f "${FINAL_REPORT}" ]]; then
      mkdir -p "${PRE_MATCHED_BACKUP}"
      [[ -d "${NVT_DIR}" ]] && cp -a "${NVT_DIR}" "${PRE_MATCHED_BACKUP}/final_nvt"
      [[ -f "${STRUCTURE_REPORT}" ]] && cp -a "${STRUCTURE_REPORT}" "${PRE_MATCHED_BACKUP}/runtime_structure_report.json"
      [[ -f "${FINAL_REPORT}" ]] && cp -a "${FINAL_REPORT}" "${PRE_MATCHED_BACKUP}/conservative_in_loop_dihedral_ibi_test_report.json"
      echo "[PROVENANCE] Archived the previous protocol-mismatched final sample under ${PRE_MATCHED_BACKUP}"
    fi
  fi
}

run_final_checks() {
  rm -rf "${NVT_DIR}"
  mkdir -p "${NVT_DIR}"

  "${PYPRESSO}" "${ROOT}/simulation/diagnose_conservative_spline_parity.py" \
    --priors "${FINAL_PRIORS}" --report "${PARITY_REPORT}"

  # The final post-update prior U_N must be sampled exactly like U_0...U_(N-1)
  # were sampled inside run_ibi_loop.py.  Derive every protocol parameter and
  # the next deterministic seeds from the persisted IBI report; do not use a
  # production checkpoint or a different dataset/model path here.
  matched_protocol_line="$("${PYTHON_BIN}" - "${IBI_REPORT}" "${FINAL_SAMPLING_REPORT}" "${TARGET_DATASET}" <<'PY'
import json
import sys
from pathlib import Path

ibi_path = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()
target_dataset = Path(sys.argv[3]).resolve()
ibi = json.loads(ibi_path.read_text())
metrics = list(ibi.get("metrics", []))
if not metrics:
    raise SystemExit("IBI report contains no completed sampling iterations")
last_iteration = max(int(row["iteration"]) for row in metrics)
next_iteration = last_iteration + 1
dt = float(ibi["dt_ps"])
burn = int(ibi["burn_in_steps"])
prod = int(ibi["production_steps"])
interval = int(ibi["sample_interval"])
kT = float(ibi["kT"])
velocity_seed = int(ibi["velocity_seed"]) + next_iteration - 1
thermostat_seed = int(ibi["thermostat_seed"]) + next_iteration - 1
neighbor = str(ibi["neighbor_search"])
protocol = {
    "schema_version": 1,
    "kind": "matched_final_ibi_sampling_protocol",
    "ibi_report": str(ibi_path),
    "source_priors": str(Path(ibi["final_priors"]).resolve()),
    "dataset": str(target_dataset),
    "starting_state": "target_dataset_initial_frame_plus_initialized_velocities",
    "sampled_iteration": next_iteration,
    "dt_ps": dt,
    "burn_in_steps": burn,
    "production_steps": prod,
    "total_steps": burn + prod,
    "sample_interval": interval,
    "kT": kT,
    "init_kT": kT,
    "velocity_seed": velocity_seed,
    "thermostat_seed": thermostat_seed,
    "neighbor_search": neighbor,
    "checkpoint_used": False,
    "ml_active": False,
    "matched_to_ibi_loop": True,
}
out_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
print(dt, burn, prod, interval, kT, velocity_seed, thermostat_seed, neighbor, next_iteration)
PY
  )"
  read -r IBI_DT IBI_BURN IBI_PROD IBI_INTERVAL IBI_KT IBI_VELOCITY_SEED IBI_THERMOSTAT_SEED IBI_NEIGHBOR IBI_FINAL_ITERATION <<EOF_PROTOCOL
${matched_protocol_line}
EOF_PROTOCOL
  for value in "${IBI_DT:-}" "${IBI_BURN:-}" "${IBI_PROD:-}" "${IBI_INTERVAL:-}" "${IBI_KT:-}" \
               "${IBI_VELOCITY_SEED:-}" "${IBI_THERMOSTAT_SEED:-}" "${IBI_NEIGHBOR:-}" "${IBI_FINAL_ITERATION:-}"; do
    [[ -n "${value}" ]] || { echo "[ERROR] Could not derive matched final-sampling protocol" >&2; exit 1; }
  done
  IBI_TOTAL=$((IBI_BURN + IBI_PROD))

  echo "[MATCHED FINAL SAMPLE] iteration=${IBI_FINAL_ITERATION} burn-in=${IBI_BURN} production=${IBI_PROD} dt=${IBI_DT} ps"
  echo "[MATCHED FINAL SAMPLE] velocity_seed=${IBI_VELOCITY_SEED} thermostat_seed=${IBI_THERMOSTAT_SEED} dataset=${TARGET_DATASET}"

  "${PYPRESSO}" "${ROOT}/simulation/run_cg_md.py" \
    --config "${CONFIG}" --priors "${FINAL_PRIORS}" --rb_info "${RB_INFO}" --dataset "${TARGET_DATASET}" \
    --dt "${IBI_DT}" --steps "${IBI_TOTAL}" --log_interval "${IBI_INTERVAL}" \
    --sample_start_step "${IBI_BURN}" --sample_npz "${NVT_SAMPLE}" \
    --kT "${IBI_KT}" --init_kT "${IBI_KT}" \
    --velocity_seed "${IBI_VELOCITY_SEED}" --thermostat_seed "${IBI_THERMOSTAT_SEED}" \
    --neighbor_search "${IBI_NEIGHBOR}" --no_log > "${NVT_DIR}/run.log" 2>&1

  echo "[PASS] Final conservative-in-loop torsional prior completed matched IBI sampling."

  "${PYTHON_BIN}" "${ROOT}/ibi/validate_runtime_structure.py" \
    --dataset "${TARGET_DATASET}" --priors "${FINAL_PRIORS}" \
    --sample-npz "${NVT_SAMPLE}" --ibi-config "${SETTINGS}" --output "${STRUCTURE_REPORT}"

  final_args=(
    "${ROOT}/simulation/finalize_conservative_dihedral_ibi_loop_test.py"
    --ibi-report "${IBI_REPORT}"
    --final-priors "${FINAL_PRIORS}"
    --parity-report "${PARITY_REPORT}"
    --structure-report "${STRUCTURE_REPORT}"
    --final-sampling-report "${FINAL_SAMPLING_REPORT}"
    --direction-flat-tolerance-l1 "${IBI_DIHEDRAL_LOOP_DIRECTION_FLAT_TOLERANCE_L1}"
    --model-config-provenance "${CONFIG_PROVENANCE}" --output "${FINAL_REPORT}"
  )
  [[ -f "${STEP35_REPORT}" ]] && final_args+=(--step35-report "${STEP35_REPORT}")
  [[ -f "${STEP37_REPORT}" ]] && final_args+=(--step37-report "${STEP37_REPORT}")
  "${PYTHON_BIN}" "${final_args[@]}"
}

if [[ "${MODE}" == "--dry-run" ]]; then
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT
  "${PYTHON_BIN}" "${ROOT}/ibi/prepare_dihedral_ibi_test_seed.py" \
    --base-priors "${BASE_PRIORS}" --output "${tmpdir}/seed.json" --report "${tmpdir}/seed_report.json" --grouping-strategy "${IBI_DIHEDRAL_LOOP_GROUPING_STRATEGY}"
  "${PYTHON_BIN}" - <<PY
import json
r=json.load(open("${tmpdir}/seed_report.json"))
print(f"[PLAN] conservative-in-loop torsions: {r['dihedral_occurrences']} occurrences in {r['unique_groups']} pooled groups")
PY
  "${PYTHON_BIN}" - "${SETTINGS}" "${ITERATIONS}" <<'PY2'
import json,sys
cfg=json.load(open(sys.argv[1])); n=int(sys.argv[2]); sim=cfg["simulation"]
per=int(sim["burn_in_steps"])+int(sim["steps"])
print(f"[PLAN] New MD work: approximately {(n+1)*per} integration steps from configured IBI sampling.")
PY2
  echo "[PLAN] Sampling sequence: conservative DBI -> ${ITERATIONS} IBI sampling/update cycles -> final matched IBI sampling."
  echo "[PLAN] No production artifact will be modified."
  exit 0
fi

if [[ "${MODE}" == "--resume" ]]; then
  for path in "${IBI_REPORT}" "${FINAL_PRIORS}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Cannot resume; missing completed IBI artifact: ${path}" >&2; exit 1; }
  done
  [[ -f "${CONFIG_PROVENANCE}" ]] || { mkdir -p "${OUT}"; write_model_dependent_provenance "${CONFIG_PROVENANCE}"; }
  echo "[RESUME] Reusing completed conservative-in-loop IBI and rerunning only final parity/matched-sampling/analysis."
  archive_pre_matched_final_sampling
  run_final_checks
  exit 0
fi

if [[ -e "${OUT}" ]]; then
  if [[ "${OVERWRITE}" != "1" ]]; then
    echo "[ERROR] Output already exists: ${OUT}" >&2
    echo "[HINT] Remove it or rerun with OVERWRITE=1." >&2
    exit 1
  fi
  rm -rf "${OUT}"
fi

mkdir -p "${OUT}"
write_model_dependent_provenance "${CONFIG_PROVENANCE}"
prepare_seed
run_ibi_loop
run_final_checks

cat <<EOF
[DONE] conservative-in-loop IBI report: ${IBI_REPORT}
[DONE] final candidate priors          : ${FINAL_PRIORS}
[DONE] final test report               : ${FINAL_REPORT}
[NOTE] Nothing was promoted. The L1 sequence is now sampled with one matched IBI protocol throughout.
EOF
