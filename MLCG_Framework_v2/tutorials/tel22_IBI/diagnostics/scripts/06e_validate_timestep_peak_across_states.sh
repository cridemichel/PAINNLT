#!/usr/bin/env bash
set -euo pipefail

# TEL22_IBI state-replicate diagnostic for the narrow dt=0.0015 ps C2 peak.
#
# Purpose:
#   - keep the promoted conservative IBI Hamiltonian unchanged (PaiNN OFF);
#   - generate two NEW NVT states from the same validated source checkpoint,
#     using independent thermostat seeds;
#   - on each new state run only dt = 0.00125, 0.00150, 0.00175, 0.00200 ps
#     for 1 ps each;
#   - compare them with the already-completed 06d reference state;
#   - determine whether the dt=0.0015 ps C2 amplification is persistent across
#     states or depends on the microscopic state/phase;
#   - check whether dt=0.002 ps remains robust across all three states.
#
# Diagnostic only. No production prior/model/checkpoint is modified or promoted.

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
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"
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
NVE_DTS="${NVE_DTS:-0.00125 0.0015 0.00175 0.002}"
NVE_DURATION_PS="${NVE_DURATION_PS:-1.0}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.8}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.2}"
NVE_MIN_R2="${NVE_MIN_R2:-0.95}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-2e-5}"

# New NVT branches. Keep the preparation protocol matched to step 06; only the
# thermostat seed changes between branches B/C.
PREP_DT="${NVE_PREP_DT:-${IBI_PROMOTION_NVT_DT}}"
PREP_STEPS="${NVE_PREP_STEPS:-${IBI_PROMOTION_NVT_STEPS}}"
PREP_KT="${NVE_PREP_KT:-${IBI_PROMOTION_NVT_KT}}"
BRANCH_B_SEED="${BRANCH_B_SEED:-360602}"
BRANCH_C_SEED="${BRANCH_C_SEED:-360603}"

OUT_ROOT="${OUT_ROOT:-diagnostics/nve/nve_ibi_timestep_state_replicates}"
REFERENCE_REPORT="${REFERENCE_REPORT:-diagnostics/nve/nve_fine_timestep_response_promoted_ibi/nve_diagnostic_report.json}"
FINAL_REPORT="${OUT_ROOT}/state_replicate_report.json"
PEAK_DT="${PEAK_DT:-0.0015}"
ROBUST_DT="${ROBUST_DT:-0.002}"
PEAK_RATIO_THRESHOLD="${PEAK_RATIO_THRESHOLD:-1.5}"
ROBUST_C2_SPREAD_MAX="${ROBUST_C2_SPREAD_MAX:-1.75}"

usage() {
    cat <<'USAGE'
Usage:
  06e_validate_timestep_peak_across_states.sh [--dry-run | --overwrite | --resume]

Runs two new independent conservative-only NVT branches and, from each, a
matched 1 ps NVE scan at:

  0.00125 0.00150 0.00175 0.00200 ps

The existing 06d state is the third/reference branch. The final report tests:
  - persistence of the narrow dt=0.0015 ps C2 peak;
  - robustness of dt=0.002 ps across all three states.

Useful overrides:
  NVE_PREP_STEPS=1000
  BRANCH_B_SEED=360602
  BRANCH_C_SEED=360603
  PEAK_RATIO_THRESHOLD=1.5
  ROBUST_C2_SPREAD_MAX=1.75
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

for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${PRIORS}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing required input: ${path}" >&2; exit 1; }
done
[[ -f "${CERTIFIER}" ]] || { echo "[ERROR] Missing certifier: ${CERTIFIER}" >&2; exit 1; }
[[ -f "${RUNNER}" ]] || { echo "[ERROR] Missing runner: ${RUNNER}" >&2; exit 1; }
[[ -f "${REFERENCE_REPORT}" ]] || {
    echo "[ERROR] Missing 06d reference report: ${REFERENCE_REPORT}" >&2
    echo "        Complete 06d_map_nve_fine_timestep_response.sh first." >&2
    exit 1
}

SOURCE_CHECKPOINT=""
for candidate in \
    "${IBI_PROMOTION_SOURCE_CHECKPOINT}" \
    diagnostics/ml/postibi_runtime_validation/equilibrated_postibi.npz \
    equilibrated.npz; do
    if [[ -f "${candidate}" ]]; then
        SOURCE_CHECKPOINT="${candidate}"
        break
    fi
done
[[ -n "${SOURCE_CHECKPOINT}" ]] || {
    echo "[ERROR] No source checkpoint found." >&2
    exit 1
}

read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 4)) || {
    echo "[ERROR] This matched replicate diagnostic expects exactly four dt values." >&2
    exit 1
}

cat <<EOF_PLAN
[TEL22_IBI NVE TIMESTEP PEAK STATE-REPLICATE DIAGNOSTIC]
priors              : ${PRIORS}
model provenance    : ${MODEL}
PaiNN dynamics      : DISABLED
source checkpoint   : ${SOURCE_CHECKPOINT}
new NVT branches    : B(seed=${BRANCH_B_SEED}), C(seed=${BRANCH_C_SEED})
NVT preparation     : ${PREP_STEPS} steps at dt=${PREP_DT} ps, kT=${PREP_KT}
NVE dt grid [ps]    : ${NVE_DTS}
NVE duration / dt   : ${NVE_DURATION_PS} ps
reference branch    : existing 06d report
peak target         : dt=${PEAK_DT} ps
robustness target   : dt=${ROBUST_DT} ps
peak threshold      : C2 / geometric-neighbor-C2 >= ${PEAK_RATIO_THRESHOLD}
robust C2 spread    : max/min across states <= ${ROBUST_C2_SPREAD_MAX}
thermostat in NVE   : OFF
ML precision flag   : ${NVE_ML_PRECISION} (inactive because PaiNN is disabled)
output              : ${OUT_ROOT}
EOF_PLAN

if [[ "${MODE}" == "dry-run" ]]; then
    per_branch_steps="$(${PYTHON_BIN} - "${NVE_DURATION_PS}" "${DT_ARGS[@]}" <<'PY'
import sys
D=float(sys.argv[1])
dts=[float(x) for x in sys.argv[2:]]
print(sum(int(round(D/dt)) for dt in dts))
PY
)"
    total_new_steps=$((2 * PREP_STEPS + 2 * per_branch_steps))
    echo "[PLAN] Branch B: fresh NVT seed=${BRANCH_B_SEED}, then ${per_branch_steps} NVE integration steps."
    echo "[PLAN] Branch C: fresh NVT seed=${BRANCH_C_SEED}, then ${per_branch_steps} NVE integration steps."
    echo "[PLAN] Total new integration steps including NVT preparation: ${total_new_steps}."
    echo "[PLAN] Production IBI priors remain untouched."
    exit 0
fi

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUT_ROOT}"
fi
mkdir -p "${OUT_ROOT}"

run_branch() {
    local label="$1"
    local seed="$2"
    local branch_dir="${OUT_ROOT}/${label}"
    local prep_dir="${branch_dir}/nvt"
    local nve_dir="${branch_dir}/nve"
    local checkpoint="${prep_dir}/equilibrated_${label}.npz"
    local energy="${prep_dir}/energy.csv"
    local log="${prep_dir}/run.log"

    mkdir -p "${prep_dir}"

    if [[ "${MODE}" != "resume" || ! -s "${checkpoint}" ]]; then
        rm -f "${checkpoint}" "${energy}" "${log}"
        echo "[RUN] ${label}: fresh independent conservative NVT (seed=${seed})"
        "${PYPRESSO}" "${RUNNER}" \
            --model "${MODEL}" \
            --disable_ml \
            --config "${CONFIG}" \
            --priors "${PRIORS}" \
            --rb_info "${RB_INFO}" \
            --dataset "${DATASET}" \
            --checkpoint "${SOURCE_CHECKPOINT}" \
            --allow_checkpoint_mismatch \
            --dt "${PREP_DT}" \
            --steps "${PREP_STEPS}" \
            --log_interval 10 \
            --device "${NVE_DEVICE}" \
            --ml_precision "${NVE_ML_PRECISION}" \
            --neighbor_search "${NVE_NEIGHBOR_SEARCH}" \
            --energy_file "${energy}" \
            --no_vtf \
            --kT "${PREP_KT}" \
            --thermostat_seed "${seed}" \
            --out_checkpoint "${checkpoint}" \
            2>&1 | tee "${log}"
    else
        echo "[REUSE] ${label}: ${checkpoint}"
    fi
    [[ -s "${checkpoint}" ]] || { echo "[ERROR] ${label}: missing NVT checkpoint ${checkpoint}" >&2; exit 1; }

    CERT_CMD=(
        "${PYTHON_BIN}" "${CERTIFIER}"
        --pypresso "${PYPRESSO}"
        --model "${MODEL}"
        --disable-ml
        --config "${CONFIG}"
        --priors "${PRIORS}"
        --rb-info "${RB_INFO}"
        --dataset "${DATASET}"
        --checkpoint "${checkpoint}"
        --require-checkpoint-hamiltonian-mode conservative_classical_model_provenance_ml_disabled
        --require-checkpoint-source "${SOURCE_CHECKPOINT}"
        --dts "${DT_ARGS[@]}"
        --duration-ps "${NVE_DURATION_PS}"
        --device "${NVE_DEVICE}"
        --ml-precision "${NVE_ML_PRECISION}"
        --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
        --output-dir "${nve_dir}"
        --slope-min "${NVE_SLOPE_MIN}"
        --slope-max "${NVE_SLOPE_MAX}"
        --min-r2 "${NVE_MIN_R2}"
        --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
    )
    if [[ "${MODE}" == "resume" ]]; then
        CERT_CMD+=(--reuse-existing)
    elif [[ "${MODE}" == "overwrite" ]]; then
        CERT_CMD+=(--overwrite)
    fi

    # This is diagnostic-only in intent, but a four-point grid cannot satisfy
    # certify_nve.py's split-fit diagnostic requirement (>=3 points per disjoint
    # regime). Run strict certification to obtain the standard report, accept
    # rc=2 as a diagnostic fit failure, and continue to the cross-state analysis.
    set +e
    "${CERT_CMD[@]}"
    local rc=$?
    set -e
    if [[ ${rc} -ne 0 && ${rc} -ne 2 ]]; then
        echo "[ERROR] ${label}: certify_nve.py failed with rc=${rc}" >&2
        exit "${rc}"
    fi
    if [[ ${rc} -eq 2 ]]; then
        echo "[NOTE] ${label}: strict global certification did not pass; continuing diagnostic comparison."
    fi
    [[ -f "${nve_dir}/nve_certification_report.json" ]] || {
        echo "[ERROR] ${label}: missing NVE report ${nve_dir}/nve_certification_report.json" >&2
        exit 1
    }
}

run_branch branch_B "${BRANCH_B_SEED}"
run_branch branch_C "${BRANCH_C_SEED}"

"${PYTHON_BIN}" - \
    "${REFERENCE_REPORT}" \
    "${OUT_ROOT}/branch_B/nve/nve_certification_report.json" \
    "${OUT_ROOT}/branch_C/nve/nve_certification_report.json" \
    "${FINAL_REPORT}" \
    "${PEAK_DT}" \
    "${ROBUST_DT}" \
    "${PEAK_RATIO_THRESHOLD}" \
    "${ROBUST_C2_SPREAD_MAX}" \
    "${NVE_MAX_RELATIVE_DRIFT}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

ref_path = Path(sys.argv[1])
b_path = Path(sys.argv[2])
c_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])
peak_dt = float(sys.argv[5])
robust_dt = float(sys.argv[6])
peak_threshold = float(sys.argv[7])
robust_spread_max = float(sys.argv[8])
drift_limit = float(sys.argv[9])


def load_runs(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = sorted(data["runs"], key=lambda x: float(x["dt_ps"]))
    out = []
    for r in runs:
        dt = float(r["dt_ps"])
        sigma = float(r["sigma_E"])
        out.append({
            "dt_ps": dt,
            "sigma_E": sigma,
            "c2": sigma / (dt * dt),
            "relative_block_mean_drift": float(r["relative_block_mean_drift"]),
        })
    return out


def select_grid(runs):
    wanted = (0.00125, 0.0015, 0.00175, 0.002)
    selected = []
    for dt in wanted:
        matches = [r for r in runs if abs(r["dt_ps"] - dt) < 1.0e-12]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one dt={dt:g} row, got {len(matches)}")
        selected.append(matches[0])
    return selected


def row_at(rows, dt):
    for r in rows:
        if abs(r["dt_ps"] - dt) < 1.0e-12:
            return r
    raise RuntimeError(f"Missing dt={dt:g}")


def peak_metrics(rows):
    ordered = sorted(rows, key=lambda x: x["dt_ps"])
    cur = row_at(ordered, peak_dt)
    idx = ordered.index(cur)
    if idx == 0 or idx == len(ordered) - 1:
        raise RuntimeError("Peak dt must have immediate neighbours in the selected grid")
    left, right = ordered[idx - 1], ordered[idx + 1]
    geom = math.sqrt(left["c2"] * right["c2"])
    ratio = cur["c2"] / geom
    med = statistics.median(r["c2"] for r in ordered)
    return {
        "peak_dt_ps": peak_dt,
        "peak_c2": cur["c2"],
        "neighbor_geometric_mean_c2": geom,
        "peak_ratio": ratio,
        "c2_over_branch_median": cur["c2"] / med,
        "strong_peak": ratio >= peak_threshold,
    }

branches = {
    "reference": select_grid(load_runs(ref_path)),
    "branch_B": select_grid(load_runs(b_path)),
    "branch_C": select_grid(load_runs(c_path)),
}

branch_summary = {}
for name, rows in branches.items():
    pm = peak_metrics(rows)
    c2s = [r["c2"] for r in rows]
    branch_summary[name] = {
        "rows": rows,
        "c2_median": statistics.median(c2s),
        "c2_spread": max(c2s) / min(c2s),
        "peak": pm,
        "max_relative_block_mean_drift": max(r["relative_block_mean_drift"] for r in rows),
    }

peak_count = sum(1 for item in branch_summary.values() if item["peak"]["strong_peak"])
if peak_count == 3:
    peak_class = "persistent_0p0015_peak_3of3"
elif peak_count >= 1:
    peak_class = f"state_dependent_0p0015_peak_{peak_count}of3"
else:
    peak_class = "no_reproducible_0p0015_peak"

robust_rows = {name: row_at(rows, robust_dt) for name, rows in branches.items()}
robust_c2 = [r["c2"] for r in robust_rows.values()]
robust_spread = max(robust_c2) / min(robust_c2)
robust_max_drift = max(r["relative_block_mean_drift"] for r in robust_rows.values())
robust_pass = robust_spread <= robust_spread_max and robust_max_drift <= drift_limit

report = {
    "purpose": "Test persistence of the narrow dt=0.0015 ps C2 amplification across independent NVT states",
    "diagnostic_only": True,
    "peak_dt_ps": peak_dt,
    "robust_dt_ps": robust_dt,
    "peak_ratio_threshold": peak_threshold,
    "robust_c2_spread_max": robust_spread_max,
    "drift_limit": drift_limit,
    "branches": branch_summary,
    "classification": {
        "strong_peak_count": peak_count,
        "peak_persistence": peak_class,
        "robust_dt_c2_spread_across_states": robust_spread,
        "robust_dt_max_drift": robust_max_drift,
        "robust_dt_pass": robust_pass,
    },
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print("\n[TEL22_IBI TIMESTEP PEAK STATE-REPLICATE SUMMARY]")
print("[STATE TABLE] branch     dt_ps     C2          drift")
for name, rows in branches.items():
    for r in rows:
        print(f"              {name:<10} {r['dt_ps']:<9g} {r['c2']:<11.6g} {r['relative_block_mean_drift']:.3e}")
for name, item in branch_summary.items():
    p = item["peak"]
    print(
        f"[PEAK] {name:<10} dt={peak_dt:g} C2={p['peak_c2']:.6g} "
        f"ratio_vs_neighbor_geom={p['peak_ratio']:.3f} strong={p['strong_peak']}"
    )
print(f"[CLASSIFY] peak_persistence={peak_class}")
print(
    f"[ROBUST dt={robust_dt:g}] C2 spread across states={robust_spread:.3f} "
    f"maxdrift={robust_max_drift:.3e} pass={robust_pass}"
)
for name, r in robust_rows.items():
    print(f"[ROBUST] {name:<10} C2={r['c2']:.6g} drift={r['relative_block_mean_drift']:.3e}")
print(f"[REPORT] {out_path}")
print("[DONE] Diagnostic only. Production IBI priors were not modified.")
PY
