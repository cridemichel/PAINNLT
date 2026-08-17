#!/usr/bin/env bash
set -euo pipefail

# TEL22_IBI duration-localization diagnostic.
#
# Purpose:
#   - keep the promoted conservative IBI Hamiltonian and the exact shared NVT
#     checkpoint produced by step 06;
#   - run a fresh matched 1 ps NVE scan on the same dt grid;
#   - re-analyze the already completed 2 ps trajectories as equal-duration
#     early/late ~1 ps windows;
#   - distinguish duration/window effects from checkpoint/state dependence.
#
# This script is diagnostic-only. It never modifies production priors and does
# not promote/reject a candidate by itself.

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

NVE_DEVICE="cpu"
NVE_ML_PRECISION="float32"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-${IBI_PROMOTION_NEIGHBOR_SEARCH}}"
NVE_DTS="${NVE_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
SHORT_DURATION_PS="${SHORT_DURATION_PS:-1.0}"
LONG_DURATION_PS="${LONG_DURATION_PS:-2.0}"

PREP_DIR="${NVE_PREP_DIR:-diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_shared_nvt}"
PREP_CHECKPOINT="${PREP_DIR}/equilibrated_promoted_ibi_recheck.npz"
LONG_DIR="${LONG_NVE_DIR:-diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_fp32}"
SHORT_DIR="${SHORT_NVE_DIR:-diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_1ps_diagnostic}"
DIAG_DIR="${DURATION_DIAG_DIR:-diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_duration_diagnostic}"
DIAG_REPORT="${DIAG_DIR}/duration_diagnostic_report.json"

# Thresholds are retained only as a common reference. certify_nve.py is called
# with --diagnostic-only, so a failed strict fit never aborts this localization.
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.8}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.2}"
NVE_MIN_R2="${NVE_MIN_R2:-0.95}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-2e-5}"

usage() {
    cat <<'USAGE'
Usage:
  06c_diagnose_nve_duration_dependence.sh [--dry-run | --overwrite | --resume]

Requires the completed 2 ps FP32 conservative-only recheck from step 06.
It reuses the exact shared NVT checkpoint, runs a fresh 1 ps scan, and compares:
  - fresh 1 ps trajectories;
  - first ~1 ps of each existing 2 ps trajectory;
  - last  ~1 ps of each existing 2 ps trajectory;
  - full 2 ps trajectories.

No production files are modified. PaiNN remains disabled.
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

for path in "${MODEL}" "${MODEL}.manifest.json" "${CONFIG}" "${DATASET}" "${RB_INFO}" "${PRIORS}" "${PREP_CHECKPOINT}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing required input: ${path}" >&2; exit 1; }
done
[[ -f "${CERTIFIER}" ]] || { echo "[ERROR] Missing certifier: ${CERTIFIER}" >&2; exit 1; }

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
[[ -n "${SOURCE_CHECKPOINT}" ]] || { echo "[ERROR] No source checkpoint found." >&2; exit 1; }

read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} >= 3)) || { echo "[ERROR] NVE_DTS needs at least three values" >&2; exit 1; }

# Verify the 2 ps source series before launching any new MD.
for dt in "${DT_ARGS[@]}"; do
    tag="$("${PYTHON_BIN}" - "${dt}" <<'PY'
import sys
print((f"dt_{float(sys.argv[1]):.8g}").replace(".", "p"))
PY
)"
    energy="${LONG_DIR}/${tag}/energy.csv"
    [[ -f "${energy}" ]] || {
        echo "[ERROR] Missing completed 2 ps energy series: ${energy}" >&2
        echo "        Run 06_test_nve_scaling_and_drift.sh first." >&2
        exit 1
    }
done

cat <<EOF_PLAN
[TEL22_IBI NVE DURATION LOCALIZATION -- DIAGNOSTIC ONLY]
priors            : ${PRIORS}
model provenance  : ${MODEL}
PaiNN dynamics    : DISABLED
shared checkpoint : ${PREP_CHECKPOINT}
dt grid [ps]      : ${NVE_DTS}
existing long run : ${LONG_DURATION_PS} ps (${LONG_DIR})
fresh short run   : ${SHORT_DURATION_PS} ps (${SHORT_DIR})
window analysis   : early/late matched ~${SHORT_DURATION_PS} ps from each long trajectory
ML precision flag : ${NVE_ML_PRECISION} (inactive because PaiNN is disabled)
neighbor search   : ${NVE_NEIGHBOR_SEARCH}
report            : ${DIAG_REPORT}
EOF_PLAN

if [[ "${MODE}" == "dry-run" ]]; then
    for dt in "${DT_ARGS[@]}"; do
        echo "[PLAN] fresh NVE dt=${dt} ps duration=${SHORT_DURATION_PS} ps from shared checkpoint"
    done
    echo "[PLAN] re-analyze existing ${LONG_DURATION_PS} ps series into early/late matched windows"
    exit 0
fi

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${SHORT_DIR}" "${DIAG_DIR}"
fi
mkdir -p "${DIAG_DIR}"

CERT_CMD=(
    "${PYTHON_BIN}" "${CERTIFIER}"
    --pypresso "${PYPRESSO}"
    --model "${MODEL}"
    --disable-ml
    --config "${CONFIG}"
    --priors "${PRIORS}"
    --rb-info "${RB_INFO}"
    --dataset "${DATASET}"
    --checkpoint "${PREP_CHECKPOINT}"
    --require-checkpoint-hamiltonian-mode conservative_classical_model_provenance_ml_disabled
    --require-checkpoint-source "${SOURCE_CHECKPOINT}"
    --dts "${DT_ARGS[@]}"
    --duration-ps "${SHORT_DURATION_PS}"
    --device "${NVE_DEVICE}"
    --ml-precision "${NVE_ML_PRECISION}"
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
    --output-dir "${SHORT_DIR}"
    --slope-min "${NVE_SLOPE_MIN}"
    --slope-max "${NVE_SLOPE_MAX}"
    --min-r2 "${NVE_MIN_R2}"
    --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
    --diagnostic-only
    --diagnostic-fine-max-dt 0.002
    --diagnostic-coarse-min-dt 0.003
)
if [[ "${MODE}" == "resume" ]]; then
    CERT_CMD+=(--reuse-existing)
elif [[ "${MODE}" == "overwrite" ]]; then
    CERT_CMD+=(--overwrite)
fi

"${CERT_CMD[@]}"

# Analyze all windows with the exact same numerical helpers used by certification.
"${PYTHON_BIN}" - \
    "${FRAMEWORK_ROOT}" \
    "${LONG_DIR}" \
    "${SHORT_DIR}" \
    "${SHORT_DURATION_PS}" \
    "${LONG_DURATION_PS}" \
    "${DIAG_REPORT}" \
    "${DT_ARGS[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np

framework_root = Path(sys.argv[1]).resolve()
long_dir = Path(sys.argv[2]).resolve()
short_dir = Path(sys.argv[3]).resolve()
short_target = float(sys.argv[4])
long_target = float(sys.argv[5])
report_path = Path(sys.argv[6]).resolve()
dts = [float(x) for x in sys.argv[7:]]

sys.path.insert(0, str(framework_root / "simulation"))
from nve_analysis import analyze_energy_series, fit_timestep_scaling, read_energy_csv


def run_tag(dt: float) -> str:
    return (f"dt_{dt:.8g}").replace(".", "p")


def with_dt(metrics, dt):
    out = dict(metrics)
    out["dt_ps"] = float(dt)
    return out


def summarize(rows):
    ordered = sorted(rows, key=lambda x: float(x["dt_ps"]))
    fit = fit_timestep_scaling(ordered)
    c2 = [float(x["sigma_E"]) / float(x["dt_ps"]) ** 2 for x in ordered]
    worst = max(ordered, key=lambda x: float(x["relative_block_mean_drift"]))
    return {
        "fit": fit,
        "c2_spread": float(max(c2) / min(c2)),
        "max_relative_block_mean_drift": float(worst["relative_block_mean_drift"]),
        "max_drift_dt_ps": float(worst["dt_ps"]),
        "runs": ordered,
    }


def subset_summary(rows, keep):
    chosen = [x for x in rows if any(abs(float(x["dt_ps"]) - k) < 1e-12 for k in keep)]
    if len(chosen) < 3:
        return {"available": False}
    out = summarize(chosen)
    out["available"] = True
    return out

full_rows = []
early_rows = []
late_rows = []
fresh_rows = []
per_dt = []

for dt in sorted(dts):
    full_csv = long_dir / run_tag(dt) / "energy.csv"
    fresh_csv = short_dir / run_tag(dt) / "energy.csv"
    t_full, e_full = read_energy_csv(full_csv)
    t_fresh, e_fresh = read_energy_csv(fresh_csv)

    expected_long_steps = int(round(long_target / dt))
    expected_long_duration = expected_long_steps * dt
    duration_tol = max(1.0e-10, 1.0e-8 * max(1.0, expected_long_duration))
    actual_long_duration = float(t_full[-1] - t_full[0])
    if t_full.size != expected_long_steps + 1 or abs(actual_long_duration - expected_long_duration) > duration_tol:
        raise RuntimeError(
            f"dt={dt:g}: existing long series is not the requested matched {long_target:g} ps run "
            f"(samples={t_full.size}, duration={actual_long_duration:.17g} ps, "
            f"expected samples={expected_long_steps + 1}, duration={expected_long_duration:.17g} ps)"
        )

    full_m = with_dt(analyze_energy_series(t_full, e_full), dt)
    fresh_m = with_dt(analyze_energy_series(t_fresh, e_fresh), dt)

    # Match certify_nve.py's rounding for a nominal short_target duration.
    n_short_steps = int(round(short_target / dt))
    needed = n_short_steps + 1
    if t_full.size < needed:
        raise RuntimeError(f"dt={dt:g}: long trajectory has too few samples for early window")
    if t_full.size < needed:
        raise RuntimeError(f"dt={dt:g}: long trajectory has too few samples for late window")

    t_early = t_full[:needed]
    e_early = e_full[:needed]
    t_late = t_full[-needed:]
    e_late = e_full[-needed:]
    early_m = with_dt(analyze_energy_series(t_early, e_early), dt)
    late_m = with_dt(analyze_energy_series(t_late, e_late), dt)

    full_rows.append(full_m)
    early_rows.append(early_m)
    late_rows.append(late_m)
    fresh_rows.append(fresh_m)

    sig_prefix = float(early_m["sigma_E"])
    sig_fresh = float(fresh_m["sigma_E"])
    sig_abs = abs(sig_prefix - sig_fresh)
    sig_rel = sig_abs / max(abs(sig_prefix), abs(sig_fresh), np.finfo(float).tiny)
    per_dt.append({
        "dt_ps": dt,
        "nominal_short_duration_ps": short_target,
        "early_actual_duration_ps": float(early_m["duration_ps"]),
        "late_actual_duration_ps": float(late_m["duration_ps"]),
        "full_actual_duration_ps": float(full_m["duration_ps"]),
        "fresh_actual_duration_ps": float(fresh_m["duration_ps"]),
        "fresh_vs_early_sigma_abs": float(sig_abs),
        "fresh_vs_early_sigma_rel": float(sig_rel),
    })

summaries = {
    "fresh_1ps": summarize(fresh_rows),
    "early_window": summarize(early_rows),
    "late_window": summarize(late_rows),
    "full_2ps": summarize(full_rows),
}

# The 0.001/0.002/0.004 backbone was unusually clean in the failed 2 ps result.
backbone = [0.001, 0.002, 0.004]
for name, rows in (
    ("fresh_1ps", fresh_rows),
    ("early_window", early_rows),
    ("late_window", late_rows),
    ("full_2ps", full_rows),
):
    summaries[name]["backbone_0p001_0p002_0p004"] = subset_summary(rows, backbone)

max_sigma_rel = max(float(x["fresh_vs_early_sigma_rel"]) for x in per_dt)
early_r2 = float(summaries["early_window"]["fit"]["loglog_r2"])
late_r2 = float(summaries["late_window"]["fit"]["loglog_r2"])
early_c2 = float(summaries["early_window"]["c2_spread"])
late_c2 = float(summaries["late_window"]["c2_spread"])

# Conservative labels only; this is localization, not a new certification gate.
if max_sigma_rel <= 1.0e-10:
    reproducibility = "fresh_1ps_matches_2ps_prefix"
else:
    reproducibility = "fresh_1ps_differs_from_2ps_prefix"

if early_r2 >= 0.95 and late_r2 < 0.95:
    localization = "late_window_degradation"
elif early_r2 < 0.95 and late_r2 < 0.95:
    localization = "both_windows_irregular"
elif early_r2 >= 0.95 and late_r2 >= 0.95:
    localization = "both_windows_scaling_clean"
else:
    localization = "early_window_worse_than_late"

report = {
    "definition": {
        "purpose": "Matched 1 ps / 2 ps duration localization for promoted conservative TEL22_IBI",
        "diagnostic_only": True,
        "same_initial_checkpoint": True,
        "painn_active": False,
        "short_target_ps": short_target,
        "long_target_ps": long_target,
        "early_late_window_rule": "first/last round(short_target/dt)+1 samples from existing long trajectory",
    },
    "paths": {
        "long_dir": str(long_dir),
        "short_dir": str(short_dir),
    },
    "summaries": summaries,
    "per_dt_reproducibility": per_dt,
    "max_fresh_vs_early_sigma_rel": max_sigma_rel,
    "classification": {
        "fresh_vs_prefix": reproducibility,
        "duration_localization": localization,
        "early_r2": early_r2,
        "late_r2": late_r2,
        "early_c2_spread": early_c2,
        "late_c2_spread": late_c2,
    },
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print("\n[TEL22_IBI NVE DURATION DIAGNOSTIC SUMMARY]")
for name, label in (
    ("fresh_1ps", "fresh 1 ps"),
    ("early_window", "early ~1 ps of 2 ps"),
    ("late_window", "late ~1 ps of 2 ps"),
    ("full_2ps", "full 2 ps"),
):
    s = summaries[name]
    f = s["fit"]
    print(
        f"[{label}] p={float(f['exponent_p']):.6f} "
        f"R2={float(f['loglog_r2']):.6f} "
        f"C2spread={float(s['c2_spread']):.3f} "
        f"maxdrift={float(s['max_relative_block_mean_drift']):.3e}"
    )

print("[WINDOW TABLE] dt_ps  fresh_C2  early_C2  late_C2  full_C2  fresh/early_sigma_rel")
for dt in sorted(dts):
    def row(rows):
        return next(x for x in rows if abs(float(x["dt_ps"]) - dt) < 1e-12)
    fr, er, lr, ar = row(fresh_rows), row(early_rows), row(late_rows), row(full_rows)
    rel = next(x["fresh_vs_early_sigma_rel"] for x in per_dt if abs(x["dt_ps"] - dt) < 1e-12)
    print(
        f"               {dt:<6g} "
        f"{float(fr['sigma_E'])/dt**2:<10.3f} "
        f"{float(er['sigma_E'])/dt**2:<10.3f} "
        f"{float(lr['sigma_E'])/dt**2:<10.3f} "
        f"{float(ar['sigma_E'])/dt**2:<10.3f} "
        f"{float(rel):.3e}"
    )

print(f"[REPRO] max fresh-vs-prefix relative sigma difference={max_sigma_rel:.3e}")
print(f"[CLASSIFY] fresh_vs_prefix={reproducibility}")
print(f"[CLASSIFY] duration_localization={localization}")
print(f"[REPORT] {report_path}")
PY

echo "[DONE] Diagnostic only. Production IBI priors were not modified."
