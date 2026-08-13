#!/usr/bin/env bash
set -euo pipefail

# Causal A/B diagnostic for the TEL22 1-2 WCA exclusion policy.
#
# This deliberately changes the runtime Hamiltonian relative to training:
# for topological 1-2 molecule pairs, WCA is kept on every virtual-site cross
# pair except the explicitly bonded site pair(s).  1-3 exclusions are unchanged.
# The run is diagnostic only; do not use its trajectory for production or NVE
# certification of the trained Hamiltonian.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi

PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"

# Resolve PYRESSO before entering the per-run output directory.  A caller may
# provide a path relative to the directory from which this wrapper is invoked
# (for example ../../espresso/build/pypresso from tutorials/tel22).
if [[ "${PYRESSO}" == */* ]]; then
    if [[ "${PYRESSO}" != /* ]]; then
        PYRESSO="$(pwd -P)/${PYRESSO}"
    fi
    if [[ ! -x "${PYRESSO}" ]]; then
        echo "[ERROR] PYRESSO is not an executable file: ${PYRESSO}" >&2
        exit 127
    fi
else
    PYRESSO_RESOLVED="$(command -v "${PYRESSO}" || true)"
    if [[ -z "${PYRESSO_RESOLVED}" ]]; then
        echo "[ERROR] PYRESSO executable not found in PATH: ${PYRESSO}" >&2
        exit 127
    fi
    PYRESSO="${PYRESSO_RESOLVED}"
fi

WCA12_TEST_DT="${WCA12_TEST_DT:-0.005}"
WCA12_TEST_DURATION_PS="${WCA12_TEST_DURATION_PS:-1.3}"
WCA12_TEST_DEVICE="${WCA12_TEST_DEVICE:-cpu}"
WCA12_TEST_OUTPUT_DIR="${WCA12_TEST_OUTPUT_DIR:-wca12_selective_ab}"
WCA12_TEST_OVERWRITE="${WCA12_TEST_OVERWRITE:-0}"

cd "${SCRIPT_DIR}"

for path in tel22_model.pt tel22_training_config.json cg_priors.json rigid_bodies_info.json tel22_dataset.bin equilibrated.npz; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

read -r STEPS ACTUAL_DURATION <<< "$(python3 - "${WCA12_TEST_DT}" "${WCA12_TEST_DURATION_PS}" <<'PY'
import sys

dt = float(sys.argv[1])
duration = float(sys.argv[2])
if dt <= 0.0 or duration <= 0.0:
    raise SystemExit("dt and duration must be positive")
steps = int(round(duration / dt))
if steps < 1:
    raise SystemExit("duration is too short for the requested dt")
print(steps, format(steps * dt, ".17g"))
PY
)"

DT_TAG="$(python3 - "${WCA12_TEST_DT}" <<'PY'
import sys
print(format(float(sys.argv[1]), ".8g").replace(".", "p"))
PY
)"
RUN_DIR="${WCA12_TEST_OUTPUT_DIR}/dt_${DT_TAG}"

if [[ -e "${RUN_DIR}" ]]; then
    if [[ "${WCA12_TEST_OVERWRITE}" == "1" ]]; then
        rm -rf "${RUN_DIR}"
    else
        echo "[ERROR] ${RUN_DIR} already exists. Set WCA12_TEST_OVERWRITE=1 to replace it." >&2
        exit 2
    fi
fi
mkdir -p "${RUN_DIR}"
RUN_DIR_ABS="$(cd "${RUN_DIR}" && pwd)"

printf '%s\n' \
    "[AB-PLAN] diagnostic selective 1-2 WCA" \
    "          dt=${WCA12_TEST_DT} ps steps=${STEPS} duration=${ACTUAL_DURATION} ps" \
    "          pypresso=${PYRESSO}" \
    "          1-2: exclude only explicitly bonded virtual-site pairs" \
    "          1-3: retain production all-sites exclusions" \
    "          output=${RUN_DIR_ABS}"

echo "[WARNING] This is a deliberately altered runtime Hamiltonian for causal diagnosis only."

set +e
(
    cd "${RUN_DIR_ABS}"
    "${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
        --model "${SCRIPT_DIR}/tel22_model.pt" \
        --config "${SCRIPT_DIR}/tel22_training_config.json" \
        --priors "${SCRIPT_DIR}/cg_priors.json" \
        --rb_info "${SCRIPT_DIR}/rigid_bodies_info.json" \
        --dataset "${SCRIPT_DIR}/tel22_dataset.bin" \
        --checkpoint "${SCRIPT_DIR}/equilibrated.npz" \
        --dt "${WCA12_TEST_DT}" \
        --steps "${STEPS}" \
        --log_interval 1 \
        --device "${WCA12_TEST_DEVICE}" \
        --nve \
        --no_vtf \
        --energy_file energy.csv \
        --diagnostic_selective_wca_12 \
        > run.log 2>&1
)
RC=$?
set -e

python3 - "${RUN_DIR_ABS}/energy.csv" "${STEPS}" "${RC}" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
requested_steps = int(sys.argv[2])
rc = int(sys.argv[3])
if not path.is_file():
    print(f"[AB-RESULT] no energy.csv produced; runner exit={rc}")
    raise SystemExit(0)

with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    print(f"[AB-RESULT] empty energy.csv; runner exit={rc}")
    raise SystemExit(0)

closest = min(rows, key=lambda row: float(row["min_dist"]))
last = rows[-1]
completed = int(last["Step"])
print(
    "[AB-RESULT] "
    f"runner_exit={rc} last_step={completed}/{requested_steps} "
    f"last_t={float(last['Time_ps']):.6f} ps "
    f"global_min_dist={float(closest['min_dist']):.6f} nm "
    f"at_t={float(closest['Time_ps']):.6f} ps "
    f"types={closest['min_pair']} pids={closest['min_pids']}"
)
if rc == 0 and completed >= requested_steps:
    print("[AB-PASS] Selective 1-2 WCA run completed the requested physical duration.")
else:
    print("[AB-FAIL] Selective 1-2 WCA run did not complete; inspect run.log.")
PY

if [[ ${RC} -ne 0 ]]; then
    echo "[AB-LOG-TAIL]"
    tail -n 35 "${RUN_DIR_ABS}/run.log" || true
fi

exit "${RC}"
