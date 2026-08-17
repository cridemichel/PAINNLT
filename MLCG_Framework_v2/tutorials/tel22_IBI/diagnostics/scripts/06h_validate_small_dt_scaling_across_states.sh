#!/usr/bin/env bash
set -euo pipefail

# TEL22_IBI final small-dt multi-state NVE scaling diagnostic.
#
# Purpose:
#   - keep the promoted conservative IBI Hamiltonian unchanged (PaiNN OFF);
#   - reuse the three existing states: reference, branch_B, branch_C;
#   - test the genuinely small-timestep regime at dt=0.0005,0.00075,0.001 ps;
#   - run each dt for the same 0.999 ps physical duration (exactly commensurate
#     with all three timesteps);
#   - fit sigma_E ~ dt^p separately for each state;
#   - apply the same fixed-effects multi-replica logic used by the earlier
#     IBI angle final validation: one common p, one intercept per state;
#   - gate on common order/R2, within-state C2 flatness and energy drift,
#     NOT on equality of absolute C2 amplitudes between microscopic states.
#
# To avoid duplicating work, the dt=0.001 ps ~1-ps trajectory is seeded from
# the already-completed 06g 2-ps run when available. certify_nve.py then
# reuses that exact prefix and computes only missing dt values.
#
# Diagnostic only. No production prior/model/checkpoint/timestep is modified.

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
SOURCE_CHECKPOINT="${IBI_PROMOTION_SOURCE_CHECKPOINT:-diagnostics/nve/nve_equilibration_conservative_ibi_only/equilibrated_conservative_ibi_only.npz}"

NVE_DEVICE="cpu"
NVE_ML_PRECISION="float32"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-${IBI_PROMOTION_NEIGHBOR_SEARCH}}"
NVE_DTS="${NVE_DTS:-0.0005 0.00075 0.001}"
NVE_DURATION_PS="${NVE_DURATION_PS:-0.999}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.8}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.2}"
NVE_MIN_R2="${NVE_MIN_R2:-0.95}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-2e-5}"

# Final multi-state gate. Defaults intentionally mirror the methodology used
# by step33, specialized to the present three-point small-dt grid.
COMMON_P_MIN="${COMMON_P_MIN:-1.8}"
COMMON_P_MAX="${COMMON_P_MAX:-2.2}"
COMMON_R2_MIN="${COMMON_R2_MIN:-0.95}"
MIN_CLEAN_DT="${MIN_CLEAN_DT:-0.00075}"
FULL_CLEAN_DT="${FULL_CLEAN_DT:-0.001}"
MIN_FULL_CLEAN_REPLICAS="${MIN_FULL_CLEAN_REPLICAS:-2}"
MEDIAN_C2_SPREAD_MAX="${MEDIAN_C2_SPREAD_MAX:-2.0}"
MAX_RELATIVE_BLOCK_DRIFT="${MAX_RELATIVE_BLOCK_DRIFT:-2e-5}"

OUT_ROOT="${OUT_ROOT:-diagnostics/nve/nve_ibi_small_dt_multistate_scaling}"
FINAL_REPORT="${OUT_ROOT}/small_dt_multistate_scaling_report.json"

REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_shared_nvt/equilibrated_promoted_ibi_recheck.npz}"
BRANCH_B_CHECKPOINT="${BRANCH_B_CHECKPOINT:-diagnostics/nve/nve_ibi_timestep_state_replicates/branch_B/nvt/equilibrated_branch_B.npz}"
BRANCH_C_CHECKPOINT="${BRANCH_C_CHECKPOINT:-diagnostics/nve/nve_ibi_timestep_state_replicates/branch_C/nvt/equilibrated_branch_C.npz}"

# 06g provides exact dt=0.001, 2-ps runs. Their first 0.999 ps can be reused as
# the dt=0.001 point of this fixed-duration scaling test.
REUSE_06G_PREFIX="${REUSE_06G_PREFIX:-1}"
SIXG_ROOT="${SIXG_ROOT:-diagnostics/nve/nve_ibi_dt_0p001_duration_state_certification}"

usage() {
    cat <<'USAGE'
Usage:
  06h_validate_small_dt_scaling_across_states.sh [--dry-run | --overwrite | --resume]

Final diagnostic-only multi-state scaling scan for conservative TEL22_IBI:

  dt = 0.0005 0.00075 0.001 ps
  duration = 0.999 ps per dt (exactly commensurate with all three dt values)
  states = reference, branch_B, branch_C
  PaiNN = disabled

The primary gate uses a fixed-effects fit

  log(sigma_E) = alpha_state + p * log(dt)

so absolute C2 amplitudes may differ between microscopic states. It checks:
  common p in [1.8, 2.2]
  within-state R2 >= 0.95
  every state clean through at least 0.00075 ps (C2 within 1.5x prefix)
  at least 2/3 states clean through 0.001 ps
  median within-state C2 spread <= 2.0x
  max relative block-mean drift <= 2e-5

When available, the first 0.999 ps of each 06g dt=0.001 run is reused exactly,
so only dt=0.0005 and 0.00075 require new dynamics.

Useful overrides:
  REUSE_06G_PREFIX=0
  COMMON_P_MIN=1.8 COMMON_P_MAX=2.2 COMMON_R2_MIN=0.95
  MIN_CLEAN_DT=0.00075 FULL_CLEAN_DT=0.001
  MIN_FULL_CLEAN_REPLICAS=2 MEDIAN_C2_SPREAD_MAX=2.0
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
            "${CERTIFIER}" "${SOURCE_CHECKPOINT}" \
            "${REFERENCE_CHECKPOINT}" "${BRANCH_B_CHECKPOINT}" "${BRANCH_C_CHECKPOINT}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing required input: ${path}" >&2; exit 1; }
done

read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} == 3)) || {
    echo "[ERROR] This final small-dt diagnostic expects exactly three dt values." >&2
    exit 1
}
"${PYTHON_BIN}" - "${NVE_DURATION_PS}" "${DT_ARGS[@]}" <<'PY'
import sys
D = float(sys.argv[1])
dts = [float(x) for x in sys.argv[2:]]
if dts != sorted(dts):
    raise SystemExit("NVE_DTS must be in ascending order")
if len(dts) != 3 or any(dt <= 0.0 for dt in dts):
    raise SystemExit("need exactly three positive dt values")
for dt in dts:
    steps = int(round(D / dt))
    if steps < 2:
        raise SystemExit("duration too short")
    actual = steps * dt
    if abs(actual - D) > max(1e-12, 1e-9 * D):
        raise SystemExit(f"duration {D} ps is not commensurate with dt={dt}: actual={actual}")
PY

cat <<EOF_PLAN
[TEL22_IBI SMALL-dt MULTI-STATE NVE SCALING -- FINAL DIAGNOSTIC]
priors                    : ${PRIORS}
model provenance          : ${MODEL}
PaiNN dynamics            : DISABLED
states                    : reference, branch_B, branch_C
dt grid [ps]              : ${NVE_DTS}
duration / dt             : ${NVE_DURATION_PS} ps
sampling                  : every integration step
thermostat                : OFF (NVE)
device                    : ${NVE_DEVICE}
ML precision flag         : ${NVE_ML_PRECISION} (inactive because PaiNN is disabled)
neighbor search           : ${NVE_NEIGHBOR_SEARCH}
common p gate             : ${COMMON_P_MIN} <= p <= ${COMMON_P_MAX}
common within-state R2    : >= ${COMMON_R2_MIN}
minimum clean dt          : ${MIN_CLEAN_DT} ps for every state
full clean dt             : ${FULL_CLEAN_DT} ps in >= ${MIN_FULL_CLEAN_REPLICAS}/3 states
median C2 spread          : <= ${MEDIAN_C2_SPREAD_MAX}x
max relative drift        : <= ${MAX_RELATIVE_BLOCK_DRIFT}
reuse 06g dt=0.001 prefix : ${REUSE_06G_PREFIX}
output                    : ${OUT_ROOT}
EOF_PLAN

if [[ "${MODE}" == "dry-run" ]]; then
    new_steps=0
    for label in reference branch_B branch_C; do
        sixg_energy="${SIXG_ROOT}/${label}/energy.csv"
        for dt in "${DT_ARGS[@]}"; do
            reuse=0
            if [[ "${REUSE_06G_PREFIX}" == "1" && "${dt}" == "0.001" && -s "${sixg_energy}" ]]; then
                reuse=1
            fi
            if (( reuse == 0 )); then
                steps="$(${PYTHON_BIN} - "${NVE_DURATION_PS}" "${dt}" <<'PY'
import sys
print(int(round(float(sys.argv[1]) / float(sys.argv[2]))))
PY
)"
                new_steps=$((new_steps + steps))
            fi
        done
    done
    echo "[PLAN] New NVE integration steps: ${new_steps}."
    if [[ "${REUSE_06G_PREFIX}" == "1" ]]; then
        echo "[PLAN] Existing 06g dt=0.001 prefixes are reused where present; missing ones fall back to a fresh 0.999-ps run."
    fi
    echo "[PLAN] No NVT regeneration. Production IBI priors/timestep remain untouched."
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

seed_06g_prefix() {
    local label="$1"
    local nve_dir="$2"
    local src="${SIXG_ROOT}/${label}/energy.csv"
    local dst_dir="${nve_dir}/dt_0p001"
    local dst="${dst_dir}/energy.csv"

    [[ "${REUSE_06G_PREFIX}" == "1" ]] || return 0
    [[ -s "${src}" ]] || {
        echo "[NOTE] ${label}: no 06g energy series at ${src}; dt=0.001 will be recomputed."
        return 0
    }
    [[ -s "${dst}" ]] && return 0

    mkdir -p "${dst_dir}"
    "${PYTHON_BIN}" - "${src}" "${dst}" "${NVE_DURATION_PS}" "0.001" <<'PY'
import csv
import math
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
duration = float(sys.argv[3])
dt = float(sys.argv[4])
steps = int(round(duration / dt))
expected = steps + 1

with src.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None or "Time_ps" not in reader.fieldnames:
        raise SystemExit(f"missing Time_ps column in {src}")
    rows = list(reader)
    fieldnames = list(reader.fieldnames)

if len(rows) < expected:
    raise SystemExit(f"{src}: only {len(rows)} data rows, need at least {expected}")
selected = rows[:expected]
times = [float(row["Time_ps"]) for row in selected]
t0 = times[0]
for i, value in enumerate(times):
    expected_t = t0 + i * dt
    if not math.isclose(value, expected_t, rel_tol=1e-8, abs_tol=1e-12):
        raise SystemExit(
            f"{src}: prefix sample {i} at t={value:.17g}, expected {expected_t:.17g}"
        )
actual = times[-1] - times[0]
if not math.isclose(actual, duration, rel_tol=1e-9, abs_tol=1e-12):
    raise SystemExit(f"{src}: prefix duration={actual:.17g}, expected {duration:.17g}")

with dst.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(selected)
PY
    echo "[SEED] ${label}: reused exact first ${NVE_DURATION_PS} ps from 06g as dt=0.001 scaling point."
}

run_state() {
    local label="$1"
    local checkpoint="$2"
    local nve_dir="${OUT_ROOT}/${label}"
    mkdir -p "${nve_dir}"

    seed_06g_prefix "${label}" "${nve_dir}"

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
        --reuse-existing
    )

    # Individual strict certification is recorded but is not the final gate:
    # the final decision uses the established fixed-effects multi-state test.
    set +e
    "${CERT_CMD[@]}"
    local rc=$?
    set -e
    if [[ ${rc} -ne 0 && ${rc} -ne 2 ]]; then
        echo "[ERROR] ${label}: certify_nve.py failed with rc=${rc}" >&2
        exit "${rc}"
    fi
    if [[ ${rc} -eq 2 ]]; then
        echo "[NOTE] ${label}: individual strict 3-point certification did not pass; continuing fixed-effects multi-state analysis."
    fi
    [[ -s "${nve_dir}/nve_certification_report.json" ]] || {
        echo "[ERROR] ${label}: missing report ${nve_dir}/nve_certification_report.json" >&2
        exit 1
    }
}

run_state reference "${REFERENCE_CHECKPOINT}"
run_state branch_B "${BRANCH_B_CHECKPOINT}"
run_state branch_C "${BRANCH_C_CHECKPOINT}"

"${PYTHON_BIN}" - \
    "${FRAMEWORK_ROOT}" \
    "${OUT_ROOT}/reference/nve_certification_report.json" \
    "${OUT_ROOT}/branch_B/nve_certification_report.json" \
    "${OUT_ROOT}/branch_C/nve_certification_report.json" \
    "${FINAL_REPORT}" \
    "${COMMON_P_MIN}" \
    "${COMMON_P_MAX}" \
    "${COMMON_R2_MIN}" \
    "${FULL_CLEAN_DT}" \
    "${MIN_CLEAN_DT}" \
    "${MIN_FULL_CLEAN_REPLICAS}" \
    "${MEDIAN_C2_SPREAD_MAX}" \
    "${MAX_RELATIVE_BLOCK_DRIFT}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

framework_root = Path(sys.argv[1]).resolve()  # retained in report provenance
report_paths = {
    "reference": Path(sys.argv[2]).resolve(),
    "branch_B": Path(sys.argv[3]).resolve(),
    "branch_C": Path(sys.argv[4]).resolve(),
}
out_path = Path(sys.argv[5]).resolve()
common_p_min = float(sys.argv[6])
common_p_max = float(sys.argv[7])
common_r2_min = float(sys.argv[8])
full_clean_dt = float(sys.argv[9])
min_clean_dt = float(sys.argv[10])
min_full_clean_replicas = int(sys.argv[11])
median_c2_spread_max = float(sys.argv[12])
max_relative_block_drift = float(sys.argv[13])


def linear_fit(x, y):
    if len(x) < 3 or len(x) != len(y):
        raise ValueError("need at least three paired points")
    xm = sum(x) / len(x)
    ym = sum(y) / len(y)
    denom = sum((v - xm) ** 2 for v in x)
    if denom <= 0.0:
        raise ValueError("degenerate x grid")
    p = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / denom
    intercept = ym - p * xm
    pred = [intercept + p * a for a in x]
    sse = sum((b - q) ** 2 for b, q in zip(y, pred))
    sst = sum((b - ym) ** 2 for b in y)
    r2 = 1.0 if sst <= sys.float_info.epsilon else 1.0 - sse / sst
    return p, intercept, r2


def largest_clean_dt(dt, c2, factor=1.5):
    ref = c2[0]
    lo, hi = ref / factor, ref * factor
    best = dt[0]
    for d, value in zip(dt[1:], c2[1:]):
        if not (lo <= value <= hi):
            break
        best = d
    return best


def fit_sigma_range(runs):
    good = sorted(runs, key=lambda row: float(row["dt_ps"]))
    if len(good) < 3:
        raise ValueError("fewer than three dt values")
    dt = [float(row["dt_ps"]) for row in good]
    sigma = [float(row["sigma_E"]) for row in good]
    if any(d <= 0.0 for d in dt) or any(s <= 0.0 or not math.isfinite(s) for s in sigma):
        raise ValueError("nonpositive/nonfinite dt or sigma")
    x = [math.log(d) for d in dt]
    y = [math.log(s) for s in sigma]
    p, logc, r2 = linear_fit(x, y)
    c2 = [s / (d * d) for d, s in zip(dt, sigma)]
    local = []
    for i in range(len(dt) - 1):
        lp = math.log(sigma[i + 1] / sigma[i]) / math.log(dt[i + 1] / dt[i])
        local.append({
            "dt_low_ps": dt[i],
            "dt_high_ps": dt[i + 1],
            "local_exponent_p": lp,
            "c2_low": c2[i],
            "c2_high": c2[i + 1],
        })
    return {
        "available": True,
        "n_points": len(dt),
        "dt_ps": dt,
        "sigma_E": sigma,
        "sigma_over_dt2": c2,
        "fit": {
            "model": "sigma_E = C * dt^p",
            "exponent_p": p,
            "prefactor_C": math.exp(logc),
            "loglog_r2": r2,
        },
        "c2_spread_max_over_min": max(c2) / min(c2),
        "adjacent_local_exponents": local,
        "max_clean_dt_factor_1p5": largest_clean_dt(dt, c2, 1.5),
        "max_clean_dt_factor_2": largest_clean_dt(dt, c2, 2.0),
    }


def fixed_effects_sigma_slope(replica_rows):
    xs, ys, names = [], [], []
    for row in replica_rows:
        sr = row["sigma_range"]
        x = [math.log(float(v)) for v in sr["dt_ps"]]
        y = [math.log(float(v)) for v in sr["sigma_E"]]
        xs.append(x)
        ys.append(y)
        names.append(row["name"])
    xdm = [[v - sum(x) / len(x) for v in x] for x in xs]
    ydm = [[v - sum(y) / len(y) for v in y] for y in ys]
    denom = sum(sum(v * v for v in x) for x in xdm)
    if denom <= sys.float_info.epsilon:
        raise ValueError("degenerate dt grid for fixed-effects fit")
    p = sum(sum(a * b for a, b in zip(x, y)) for x, y in zip(xdm, ydm)) / denom
    sse = 0.0
    sst = 0.0
    intercepts = {}
    for name, x, y in zip(names, xs, ys):
        alpha = sum(y) / len(y) - p * sum(x) / len(x)
        intercepts[name] = alpha
        pred = [alpha + p * v for v in x]
        ym = sum(y) / len(y)
        sse += sum((a - b) ** 2 for a, b in zip(y, pred))
        sst += sum((a - ym) ** 2 for a in y)
    r2 = 1.0 if sst <= sys.float_info.epsilon else 1.0 - sse / sst
    return {
        "model": "log(sigma_E) = alpha_state + p * log(dt)",
        "exponent_p": p,
        "within_replica_r2": r2,
        "replica_log_intercepts": intercepts,
        "n_replicas": len(replica_rows),
        "n_points": sum(len(x) for x in xs),
    }


replica_rows = []
branches = {}
for name, path in report_paths.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = sorted(data["runs"], key=lambda row: float(row["dt_ps"]))
    sigma_range = fit_sigma_range(runs)
    max_drift = max(abs(float(row["relative_block_mean_drift"])) for row in runs)
    strict = data.get("certification", {})
    replica_rows.append({
        "name": name,
        "sigma_range": sigma_range,
        "max_relative_block_drift": max_drift,
    })
    branches[name] = {
        "source_report": str(path),
        "strict_certification_pass": bool(strict.get("pass", False)),
        "strict_scaling": strict.get("scaling"),
        "sigma_range": sigma_range,
        "max_relative_block_drift": max_drift,
    }

common = fixed_effects_sigma_slope(replica_rows)
clean = [float(row["sigma_range"]["max_clean_dt_factor_1p5"]) for row in replica_rows]
spreads = [float(row["sigma_range"]["c2_spread_max_over_min"]) for row in replica_rows]
drifts = [float(row["max_relative_block_drift"]) for row in replica_rows]
n_full = sum(value >= full_clean_dt - 1e-15 for value in clean)
checks = {
    "common_p": common_p_min <= float(common["exponent_p"]) <= common_p_max,
    "common_r2": float(common["within_replica_r2"]) >= common_r2_min,
    "enough_full_clean_replicas": n_full >= min_full_clean_replicas,
    "all_replicas_min_clean_dt": min(clean) >= min_clean_dt - 1e-15,
    "median_c2_spread": float(statistics.median(spreads)) <= median_c2_spread_max,
    "max_relative_block_drift": max(drifts) <= max_relative_block_drift,
}
gate = {
    "pass": all(checks.values()),
    "checks": checks,
    "common_fit": common,
    "clean_dt_factor_1p5_ps": clean,
    "n_full_clean_replicas": n_full,
    "c2_spreads": spreads,
    "median_c2_spread": float(statistics.median(spreads)),
    "max_relative_block_drifts": drifts,
    "thresholds": {
        "common_p_min": common_p_min,
        "common_p_max": common_p_max,
        "common_r2_min": common_r2_min,
        "full_clean_dt_ps": full_clean_dt,
        "min_clean_dt_ps": min_clean_dt,
        "min_full_clean_replicas": min_full_clean_replicas,
        "median_c2_spread_max": median_c2_spread_max,
        "max_relative_block_drift": max_relative_block_drift,
    },
}

failed = [name for name, ok in checks.items() if not ok]
result = {
    "schema_version": 1,
    "purpose": "final_small_dt_multistate_scaling_diagnostic_only",
    "framework_root": str(framework_root),
    "interpretation": (
        "Tests sigma_E~dt^p within each microscopic state and a common fixed-effects slope; "
        "absolute C2 equality across states is intentionally not a gate."
    ),
    "branches": branches,
    "multistate_gate": gate,
    "pass": bool(gate["pass"]),
    "failed_checks": failed,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

print("\n[TEL22_IBI SMALL-dt MULTI-STATE SCALING SUMMARY]")
print("[STATE TABLE] branch      p           R2          C2spread   clean1.5x_ps   maxdrift    strict")
for name in ("reference", "branch_B", "branch_C"):
    row = branches[name]
    sr = row["sigma_range"]
    fit = sr["fit"]
    print(
        f"              {name:<10} "
        f"{float(fit['exponent_p']):<11.6f} "
        f"{float(fit['loglog_r2']):<11.6f} "
        f"{float(sr['c2_spread_max_over_min']):<10.3f} "
        f"{float(sr['max_clean_dt_factor_1p5']):<15.6g} "
        f"{float(row['max_relative_block_drift']):<11.3e} "
        f"{row['strict_certification_pass']}"
    )
    vals = ", ".join(
        f"dt={float(dt):.6g}:C2={float(c2):.6g}"
        for dt, c2 in zip(sr["dt_ps"], sr["sigma_over_dt2"])
    )
    print(f"[C2] {name}: {vals}")

print(
    f"[MULTISTATE] common p={float(common['exponent_p']):.6f} "
    f"R2within={float(common['within_replica_r2']):.6f} "
    f"full-clean={n_full}/3 "
    f"medianC2spread={float(gate['median_c2_spread']):.3f} "
    f"maxdrift={max(drifts):.3e} "
    f"pass={gate['pass']}"
)
for name, ok in checks.items():
    print(f"[CHECK] {name}={ok}")
print(f"[CLASSIFY] small_dt_multistate_scaling_pass={gate['pass']}")
if gate["pass"]:
    print("[CLASSIFY] small_dt_asymptotic_regime_validated__ibi_numerical_baseline_can_be_frozen")
else:
    print("[CLASSIFY] small_dt_asymptotic_regime_not_yet_validated__" + "__".join(failed))
print(f"[REPORT] {out_path}")
print("[DONE] Diagnostic only. Production IBI priors and timestep settings were not modified.")
PY
