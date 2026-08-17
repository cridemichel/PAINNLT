#!/usr/bin/env bash
set -euo pipefail

# TEL22_IBI final duration/state robustness diagnostic for the candidate
# production timestep dt=0.00175 ps.
#
# Uses the three already-established microscopic states:
#   reference : shared checkpoint used by 06d
#   branch_B  : independent NVT state created by 06e
#   branch_C  : independent NVT state created by 06e
#
# For each state, run one conservative-only NVE trajectory at dt=0.00175 ps
# for 2 ps, then analyze:
#   - first ~1 ps window;
#   - second ~1 ps window;
#   - full ~2 ps trajectory;
#   - reproducibility of the first window against the corresponding 06e/06d
#     1 ps result.
#
# Diagnostic only. This script never modifies/promotes production priors,
# models, checkpoints, or a production timestep setting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/model_config.sh" ]]; then
    TUTORIAL_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../../model_config.sh" ]]; then
    TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    echo "[ERROR] Cannot locate tutorials/tel22_IBI from ${SCRIPT_DIR}" >&2
    exit 1
fi
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
RUNNER="${FRAMEWORK_ROOT}/simulation/run_cg_md.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"

cd "${TUTORIAL_DIR}"
source ./model_config.sh
load_model_dependent_config step34

MODEL="${IBI_MODEL}"
CONFIG="${IBI_PROMOTION_CONFIG}"
DATASET="${IBI_PROMOTION_DATASET}"
RB_INFO="${IBI_PROMOTION_RB_INFO}"
PRIORS="${IBI_PROMOTION_CURRENT_DIR}/cg_priors.json"

NVE_DEVICE="cpu"
NVE_ML_PRECISION="float32"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-${IBI_PROMOTION_NEIGHBOR_SEARCH}}"
CANDIDATE_DT="${CANDIDATE_DT:-0.00175}"
LONG_DURATION_PS="${LONG_DURATION_PS:-2.0}"
WINDOW_DURATION_PS="${WINDOW_DURATION_PS:-1.0}"
MAX_RELATIVE_DRIFT="${MAX_RELATIVE_DRIFT:-2e-5}"
MAX_WINDOW_C2_SPREAD="${MAX_WINDOW_C2_SPREAD:-1.75}"
MAX_FULL_C2_SPREAD="${MAX_FULL_C2_SPREAD:-1.75}"
MAX_WITHIN_BRANCH_WINDOW_RATIO="${MAX_WITHIN_BRANCH_WINDOW_RATIO:-1.75}"
PREFIX_MATCH_RTOL="${PREFIX_MATCH_RTOL:-1e-10}"

OUT_ROOT="${OUT_ROOT:-diagnostics/nve/nve_ibi_dt_0p00175_duration_state_certification}"
STATE_REPORT="${STATE_REPORT:-diagnostics/nve/nve_ibi_timestep_state_replicates/state_replicate_report.json}"
FINAL_REPORT="${OUT_ROOT}/dt_0p00175_duration_state_report.json"

REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_shared_nvt/equilibrated_promoted_ibi_recheck.npz}"
BRANCH_B_CHECKPOINT="${BRANCH_B_CHECKPOINT:-diagnostics/nve/nve_ibi_timestep_state_replicates/branch_B/nvt/equilibrated_branch_B.npz}"
BRANCH_C_CHECKPOINT="${BRANCH_C_CHECKPOINT:-diagnostics/nve/nve_ibi_timestep_state_replicates/branch_C/nvt/equilibrated_branch_C.npz}"

usage() {
    cat <<'USAGE'
Usage:
  06f_certify_candidate_dt_0p00175_across_states.sh [--dry-run | --overwrite | --resume]

Runs the candidate dt=0.00175 ps for 2 ps from the three states already used
by 06e (reference, branch_B, branch_C), with PaiNN disabled. The report compares:
  - first ~1 ps vs second ~1 ps in each branch;
  - full ~2 ps C2 across states;
  - first ~1 ps against the prior 06e/06d 1 ps result for exact reproducibility.

Diagnostic-only acceptance defaults:
  max C2 spread across all six 1 ps windows : 1.75x
  max full-2ps C2 spread across states      : 1.75x
  max early/late C2 ratio within a branch   : 1.75x
  max relative block-mean drift             : 2e-5

Useful overrides:
  CANDIDATE_DT=0.00175
  LONG_DURATION_PS=2.0
  WINDOW_DURATION_PS=1.0
  MAX_WINDOW_C2_SPREAD=1.75
  MAX_FULL_C2_SPREAD=1.75
  MAX_WITHIN_BRANCH_WINDOW_RATIO=1.75
USAGE
}

MODE="normal"
case "${1:-}" in
    "") ;;
    --dry-run) MODE="dry-run"; shift ;;
    --overwrite) MODE="overwrite"; shift ;;
    --resume) MODE="resume"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
esac
if (($# != 0)); then
    echo "[ERROR] Unexpected extra arguments: $*" >&2
    exit 2
fi

for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${PRIORS}" \
            "${RUNNER}" "${STATE_REPORT}" "${REFERENCE_CHECKPOINT}" "${BRANCH_B_CHECKPOINT}" "${BRANCH_C_CHECKPOINT}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing required input: ${path}" >&2; exit 1; }
done

read -r LONG_STEPS WINDOW_STEPS ACTUAL_LONG_PS ACTUAL_WINDOW_PS <<EOF_STEPS
$("${PYTHON_BIN}" - "${CANDIDATE_DT}" "${LONG_DURATION_PS}" "${WINDOW_DURATION_PS}" <<'PY'
import sys
dt = float(sys.argv[1])
long_ps = float(sys.argv[2])
window_ps = float(sys.argv[3])
long_steps = int(round(long_ps / dt))
window_steps = int(round(window_ps / dt))
if long_steps < 2 or window_steps < 2:
    raise SystemExit("durations are too short")
if 2 * (window_steps + 1) > long_steps + 1:
    raise SystemExit("two requested analysis windows do not fit in the long trajectory")
print(long_steps, window_steps, f"{long_steps * dt:.17g}", f"{window_steps * dt:.17g}")
PY
)
EOF_STEPS

cat <<EOF_PLAN
[TEL22_IBI dt=0.00175 DURATION/STATE ROBUSTNESS -- DIAGNOSTIC ONLY]
priors                 : ${PRIORS}
model provenance       : ${MODEL}
PaiNN dynamics         : DISABLED
candidate dt [ps]      : ${CANDIDATE_DT}
requested duration     : ${LONG_DURATION_PS} ps
actual duration        : ${ACTUAL_LONG_PS} ps (${LONG_STEPS} steps)
window target          : ${WINDOW_DURATION_PS} ps
actual window          : ${ACTUAL_WINDOW_PS} ps (${WINDOW_STEPS} steps)
states                  : reference, branch_B, branch_C
sampling                : every integration step
thermostat              : OFF (NVE)
device                  : ${NVE_DEVICE}
ML precision flag       : ${NVE_ML_PRECISION} (inactive because PaiNN is disabled)
neighbor search         : ${NVE_NEIGHBOR_SEARCH}
max all-window C2 spread: ${MAX_WINDOW_C2_SPREAD}x
max full C2 spread      : ${MAX_FULL_C2_SPREAD}x
max branch early/late   : ${MAX_WITHIN_BRANCH_WINDOW_RATIO}x
max drift               : ${MAX_RELATIVE_DRIFT}
prior 1ps metrics       : ${STATE_REPORT}
output                  : ${OUT_ROOT}
EOF_PLAN

if [[ "${MODE}" == "dry-run" ]]; then
    echo "[PLAN] reference: ${LONG_STEPS} NVE steps from ${REFERENCE_CHECKPOINT}"
    echo "[PLAN] branch_B : ${LONG_STEPS} NVE steps from ${BRANCH_B_CHECKPOINT}"
    echo "[PLAN] branch_C : ${LONG_STEPS} NVE steps from ${BRANCH_C_CHECKPOINT}"
    echo "[PLAN] Total new NVE integration steps: $((3 * LONG_STEPS))."
    echo "[PLAN] No NVT regeneration; existing 06e states are reused."
    echo "[PLAN] Production IBI priors and timestep settings remain untouched."
    exit 0
fi

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ROOT}"
elif [[ "${MODE}" == "normal" && -d "${OUT_ROOT}" && -n "$(ls -A "${OUT_ROOT}" 2>/dev/null || true)" ]]; then
    echo "[ERROR] Output directory is not empty: ${OUT_ROOT}" >&2
    echo "        Use --resume to reuse completed runs or --overwrite to restart." >&2
    exit 1
fi
mkdir -p "${OUT_ROOT}"

run_state() {
    local label="$1"
    local checkpoint="$2"
    local run_dir="${OUT_ROOT}/${label}"
    local energy="${run_dir}/energy.csv"
    local log="${run_dir}/run.log"

    mkdir -p "${run_dir}"
    if [[ "${MODE}" == "resume" && -s "${energy}" ]]; then
        echo "[REUSE] ${label}: ${energy}"
        return 0
    fi

    if [[ "${MODE}" == "resume" ]]; then
        rm -f "${energy}" "${log}"
    fi

    echo "[RUN] ${label}: dt=${CANDIDATE_DT} ps steps=${LONG_STEPS} duration=${ACTUAL_LONG_PS} ps"
    "${PYPRESSO}" "${RUNNER}" \
        --model "${MODEL}" \
        --disable_ml \
        --config "${CONFIG}" \
        --priors "${PRIORS}" \
        --rb_info "${RB_INFO}" \
        --dataset "${DATASET}" \
        --checkpoint "${checkpoint}" \
        --dt "${CANDIDATE_DT}" \
        --steps "${LONG_STEPS}" \
        --log_interval 1 \
        --device "${NVE_DEVICE}" \
        --ml_precision "${NVE_ML_PRECISION}" \
        --neighbor_search "${NVE_NEIGHBOR_SEARCH}" \
        --nve \
        --no_vtf \
        --energy_file "${energy}" \
        2>&1 | tee "${log}"

    [[ -s "${energy}" ]] || { echo "[ERROR] ${label}: missing energy series ${energy}" >&2; exit 1; }
}

run_state reference "${REFERENCE_CHECKPOINT}"
run_state branch_B "${BRANCH_B_CHECKPOINT}"
run_state branch_C "${BRANCH_C_CHECKPOINT}"

"${PYTHON_BIN}" - \
    "${FRAMEWORK_ROOT}" \
    "${OUT_ROOT}" \
    "${STATE_REPORT}" \
    "${FINAL_REPORT}" \
    "${CANDIDATE_DT}" \
    "${LONG_DURATION_PS}" \
    "${WINDOW_DURATION_PS}" \
    "${MAX_WINDOW_C2_SPREAD}" \
    "${MAX_FULL_C2_SPREAD}" \
    "${MAX_WITHIN_BRANCH_WINDOW_RATIO}" \
    "${MAX_RELATIVE_DRIFT}" \
    "${PREFIX_MATCH_RTOL}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np

framework_root = Path(sys.argv[1]).resolve()
out_root = Path(sys.argv[2]).resolve()
state_report_path = Path(sys.argv[3]).resolve()
final_report_path = Path(sys.argv[4]).resolve()
dt = float(sys.argv[5])
long_target = float(sys.argv[6])
window_target = float(sys.argv[7])
window_spread_limit = float(sys.argv[8])
full_spread_limit = float(sys.argv[9])
within_branch_limit = float(sys.argv[10])
drift_limit = float(sys.argv[11])
prefix_rtol = float(sys.argv[12])

sys.path.insert(0, str(framework_root / "simulation"))
from nve_analysis import analyze_energy_series, read_energy_csv

branches = ("reference", "branch_B", "branch_C")
prior = json.loads(state_report_path.read_text(encoding="utf-8"))


def prior_row(branch):
    rows = prior["branches"][branch]["rows"]
    matches = [r for r in rows if abs(float(r["dt_ps"]) - dt) < 1.0e-12]
    if len(matches) != 1:
        raise RuntimeError(f"{branch}: expected one prior dt={dt:g} row, got {len(matches)}")
    return matches[0]


def c2(metrics):
    return float(metrics["sigma_E"]) / (dt * dt)


def drift(metrics):
    return float(metrics["relative_block_mean_drift"])

long_steps = int(round(long_target / dt))
window_steps = int(round(window_target / dt))
expected_samples = long_steps + 1
needed = window_steps + 1
expected_long_duration = long_steps * dt
expected_window_duration = window_steps * dt

summary = {}
all_window_c2 = []
full_c2_values = []
all_drifts = []
prefix_rel_values = []
within_ratios = []

for branch in branches:
    energy_path = out_root / branch / "energy.csv"
    t, e = read_energy_csv(energy_path)
    actual_duration = float(t[-1] - t[0])
    tol = max(1.0e-10, 1.0e-8 * max(1.0, expected_long_duration))
    if t.size != expected_samples or abs(actual_duration - expected_long_duration) > tol:
        raise RuntimeError(
            f"{branch}: unexpected series length/duration: samples={t.size}, duration={actual_duration:.17g}; "
            f"expected samples={expected_samples}, duration={expected_long_duration:.17g}"
        )
    if 2 * needed > t.size:
        raise RuntimeError(f"{branch}: trajectory too short for two disjoint windows")

    t_early, e_early = t[:needed], e[:needed]
    t_late, e_late = t[-needed:], e[-needed:]
    full = dict(analyze_energy_series(t, e))
    early = dict(analyze_energy_series(t_early, e_early))
    late = dict(analyze_energy_series(t_late, e_late))

    prior_item = prior_row(branch)
    prior_sigma = float(prior_item["sigma_E"])
    early_sigma = float(early["sigma_E"])
    prefix_rel = abs(early_sigma - prior_sigma) / max(abs(early_sigma), abs(prior_sigma), np.finfo(float).tiny)

    early_c2 = c2(early)
    late_c2 = c2(late)
    full_c2 = c2(full)
    within_ratio = max(early_c2, late_c2) / min(early_c2, late_c2)

    all_window_c2.extend((early_c2, late_c2))
    full_c2_values.append(full_c2)
    all_drifts.extend((drift(early), drift(late), drift(full)))
    prefix_rel_values.append(prefix_rel)
    within_ratios.append(within_ratio)

    summary[branch] = {
        "energy_csv": str(energy_path),
        "actual_full_duration_ps": float(full["duration_ps"]),
        "actual_early_duration_ps": float(early["duration_ps"]),
        "actual_late_duration_ps": float(late["duration_ps"]),
        "prior_1ps": {
            "sigma_E": prior_sigma,
            "c2": float(prior_item["c2"]),
            "relative_block_mean_drift": float(prior_item["relative_block_mean_drift"]),
        },
        "early_1ps": {**early, "c2": early_c2},
        "late_1ps": {**late, "c2": late_c2},
        "full_2ps": {**full, "c2": full_c2},
        "prefix_vs_prior_sigma_rel": float(prefix_rel),
        "early_late_c2_ratio": float(within_ratio),
    }

window_spread = max(all_window_c2) / min(all_window_c2)
full_spread = max(full_c2_values) / min(full_c2_values)
max_within_ratio = max(within_ratios)
max_drift = max(all_drifts)
max_prefix_rel = max(prefix_rel_values)
prefix_match = max_prefix_rel <= prefix_rtol

candidate_pass = (
    prefix_match
    and window_spread <= window_spread_limit
    and full_spread <= full_spread_limit
    and max_within_ratio <= within_branch_limit
    and max_drift <= drift_limit
)

if candidate_pass:
    classification = "dt_0p00175_state_and_duration_robust_candidate"
else:
    failures = []
    if not prefix_match:
        failures.append("prefix_reproducibility")
    if window_spread > window_spread_limit:
        failures.append("cross_state_window_c2_spread")
    if full_spread > full_spread_limit:
        failures.append("full_2ps_c2_spread")
    if max_within_ratio > within_branch_limit:
        failures.append("within_branch_duration_dependence")
    if max_drift > drift_limit:
        failures.append("energy_drift")
    classification = "dt_0p00175_not_yet_robust__" + "__".join(failures)

report = {
    "purpose": "Final diagnostic of dt=0.00175 ps state/duration robustness for promoted conservative TEL22_IBI",
    "diagnostic_only": True,
    "painn_active": False,
    "candidate_dt_ps": dt,
    "requested_long_duration_ps": long_target,
    "actual_long_duration_ps": expected_long_duration,
    "requested_window_duration_ps": window_target,
    "actual_window_duration_ps": expected_window_duration,
    "criteria": {
        "max_window_c2_spread": window_spread_limit,
        "max_full_c2_spread": full_spread_limit,
        "max_within_branch_early_late_c2_ratio": within_branch_limit,
        "max_relative_block_mean_drift": drift_limit,
        "prefix_match_rtol": prefix_rtol,
    },
    "branches": summary,
    "aggregate": {
        "all_1ps_window_c2_spread": float(window_spread),
        "full_2ps_c2_spread_across_states": float(full_spread),
        "max_within_branch_early_late_c2_ratio": float(max_within_ratio),
        "max_relative_block_mean_drift": float(max_drift),
        "max_prefix_vs_prior_sigma_rel": float(max_prefix_rel),
        "prefix_reproducibility_pass": bool(prefix_match),
        "candidate_pass": bool(candidate_pass),
        "classification": classification,
    },
}
final_report_path.parent.mkdir(parents=True, exist_ok=True)
final_report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print("\n[TEL22_IBI dt=0.00175 DURATION/STATE ROBUSTNESS SUMMARY]")
print("[WINDOW TABLE] branch      prior_C2    early_C2    late_C2     full_C2     early/late   drift_full   prefix_rel")
for branch in branches:
    s = summary[branch]
    print(
        f"               {branch:<11} "
        f"{s['prior_1ps']['c2']:<11.3f} "
        f"{s['early_1ps']['c2']:<11.3f} "
        f"{s['late_1ps']['c2']:<11.3f} "
        f"{s['full_2ps']['c2']:<11.3f} "
        f"{s['early_late_c2_ratio']:<12.3f} "
        f"{s['full_2ps']['relative_block_mean_drift']:<12.3e} "
        f"{s['prefix_vs_prior_sigma_rel']:.3e}"
    )
print(f"[REPRO] max prior-1ps vs new-prefix sigma relative difference={max_prefix_rel:.3e} pass={prefix_match}")
print(f"[ROBUST] all six 1ps-window C2 spread={window_spread:.3f} limit={window_spread_limit:.3f}")
print(f"[ROBUST] full 2ps C2 spread across states={full_spread:.3f} limit={full_spread_limit:.3f}")
print(f"[ROBUST] max within-branch early/late C2 ratio={max_within_ratio:.3f} limit={within_branch_limit:.3f}")
print(f"[ROBUST] max relative block-mean drift={max_drift:.3e} limit={drift_limit:.3e}")
print(f"[CLASSIFY] candidate_dt_0p00175_pass={candidate_pass}")
print(f"[CLASSIFY] {classification}")
print(f"[REPORT] {final_report_path}")
print("[DONE] Diagnostic only. Production IBI priors and timestep settings were not modified.")
PY
