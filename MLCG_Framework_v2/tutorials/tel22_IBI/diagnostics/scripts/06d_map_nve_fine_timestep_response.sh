#!/usr/bin/env bash
set -euo pipefail

# TEL22_IBI fine timestep-response diagnostic.
#
# Purpose:
#   map C2(dt) = sigma_E / dt^2 on a targeted low-dt grid using the exact shared
#   conservative-only checkpoint created by step 06.  This is intended to
#   localize timestep-selective amplification from the residual stiff angular
#   modes without modifying or retuning the promoted IBI priors.
#
# Diagnostic only: no production file is changed and no candidate is promoted
# or rejected by this script.

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
NVE_DTS="${NVE_DTS:-0.001 0.00125 0.0015 0.00175 0.002 0.0025 0.003}"
NVE_DURATION_PS="${NVE_DURATION_PS:-1.0}"
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_fine_timestep_response_promoted_ibi}"
ANALYSIS_REPORT="${NVE_OUTPUT_DIR}/fine_timestep_response_report.json"

# Same shared checkpoint used by steps 06/06c: no new NVT is generated here.
PREP_DIR="${NVE_PREP_DIR:-diagnostics/nve/nve_scaling_drift_recheck_promoted_ibi_shared_nvt}"
PREP_CHECKPOINT="${PREP_DIR}/equilibrated_promoted_ibi_recheck.npz"

# Reference thresholds are printed for context only.  The certifier runs in
# diagnostic-only mode; the dense C2(dt) map is the primary output.
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.8}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.2}"
NVE_MIN_R2="${NVE_MIN_R2:-0.95}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-2e-5}"
C2_NEAR_FACTOR="${C2_NEAR_FACTOR:-1.25}"
C2_CLEAN_FACTOR="${C2_CLEAN_FACTOR:-1.5}"

usage() {
    cat <<'USAGE'
Usage:
  06d_map_nve_fine_timestep_response.sh [--dry-run | --overwrite | --resume]

Targeted low-dt diagnostic for the promoted conservative IBI Hamiltonian.
It reuses the shared checkpoint from step 06 and scans, by default:

  0.001 0.00125 0.0015 0.00175 0.002 0.0025 0.003 ps

at 1 ps per dt. PaiNN remains disabled.

Useful overrides:
  NVE_DURATION_PS=2.0
  NVE_DTS="0.001 0.00125 0.0015 0.00175 0.002 0.0025 0.003"
  C2_NEAR_FACTOR=1.25
  C2_CLEAN_FACTOR=1.5
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

cat <<EOF_PLAN
[TEL22_IBI NVE FINE TIMESTEP RESPONSE -- DIAGNOSTIC ONLY]
priors            : ${PRIORS}
model provenance  : ${MODEL}
PaiNN dynamics    : DISABLED
shared checkpoint : ${PREP_CHECKPOINT}
dt grid [ps]      : ${NVE_DTS}
duration / dt     : ${NVE_DURATION_PS} ps
sampling          : every integration step
thermostat        : OFF (NVE)
device            : ${NVE_DEVICE}
ML precision flag : ${NVE_ML_PRECISION} (inactive because PaiNN is disabled)
neighbor search   : ${NVE_NEIGHBOR_SEARCH}
reference gates   : ${NVE_SLOPE_MIN} <= p <= ${NVE_SLOPE_MAX}; R2 >= ${NVE_MIN_R2}; drift <= ${NVE_MAX_RELATIVE_DRIFT}
C2 bands          : near <= ${C2_NEAR_FACTOR}x median; clean <= ${C2_CLEAN_FACTOR}x median
output            : ${NVE_OUTPUT_DIR}
EOF_PLAN

if [[ "${MODE}" == "dry-run" ]]; then
    for dt in "${DT_ARGS[@]}"; do
        steps="$(${PYTHON_BIN} - "${dt}" "${NVE_DURATION_PS}" <<'PY'
import math, sys
print(int(round(float(sys.argv[2]) / float(sys.argv[1]))))
PY
)"
        echo "[PLAN] dt=${dt} ps ~${steps} steps for ${NVE_DURATION_PS} ps"
    done
    echo "[PLAN] No NVT rebuild; production IBI priors remain untouched."
    exit 0
fi

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${NVE_OUTPUT_DIR}"
fi

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
    --duration-ps "${NVE_DURATION_PS}"
    --device "${NVE_DEVICE}"
    --ml-precision "${NVE_ML_PRECISION}"
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
    --output-dir "${NVE_OUTPUT_DIR}"
    --slope-min "${NVE_SLOPE_MIN}"
    --slope-max "${NVE_SLOPE_MAX}"
    --min-r2 "${NVE_MIN_R2}"
    --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
    --diagnostic-only
    --diagnostic-fine-max-dt 0.00175
    --diagnostic-coarse-min-dt 0.002
)
if [[ "${MODE}" == "resume" ]]; then
    CERT_CMD+=(--reuse-existing)
elif [[ "${MODE}" == "overwrite" ]]; then
    CERT_CMD+=(--overwrite)
fi

"${CERT_CMD[@]}"

# certify_nve.py writes nve_diagnostic_report.json in diagnostic-only mode.
CERT_REPORT="${NVE_OUTPUT_DIR}/nve_diagnostic_report.json"
[[ -f "${CERT_REPORT}" ]] || {
    echo "[ERROR] Missing diagnostic report: ${CERT_REPORT}" >&2
    exit 1
}

"${PYTHON_BIN}" - \
    "${CERT_REPORT}" \
    "${ANALYSIS_REPORT}" \
    "${C2_NEAR_FACTOR}" \
    "${C2_CLEAN_FACTOR}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
near_factor = float(sys.argv[3])
clean_factor = float(sys.argv[4])
report = json.loads(src.read_text(encoding="utf-8"))
runs = sorted(report["runs"], key=lambda x: float(x["dt_ps"]))

rows = []
for r in runs:
    dt = float(r["dt_ps"])
    sigma = float(r["sigma_E"])
    drift = float(r["relative_block_mean_drift"])
    c2 = sigma / (dt * dt)
    rows.append({"dt_ps": dt, "sigma_E": sigma, "c2": c2, "relative_block_mean_drift": drift})

c2_values = [r["c2"] for r in rows]
median_c2 = statistics.median(c2_values)
min_c2 = min(c2_values)
max_c2 = max(c2_values)
spread = max_c2 / min_c2 if min_c2 > 0.0 else math.inf

for r in rows:
    r["c2_over_median"] = r["c2"] / median_c2
    if r["c2_over_median"] <= near_factor:
        r["band"] = "near_plateau"
    elif r["c2_over_median"] <= clean_factor:
        r["band"] = "clean_1p5x"
    else:
        r["band"] = "amplified"

# Local peak: a point above both immediate neighbours.  A stronger peak is also
# > near_factor times the geometric mean of its neighbours.
local_peaks = []
for i in range(1, len(rows) - 1):
    left, cur, right = rows[i - 1], rows[i], rows[i + 1]
    if cur["c2"] > left["c2"] * (1.0 + 1.0e-12) and cur["c2"] > right["c2"] * (1.0 + 1.0e-12):
        geom = math.sqrt(left["c2"] * right["c2"])
        ratio = cur["c2"] / geom if geom > 0.0 else math.inf
        local_peaks.append({
            "dt_ps": cur["dt_ps"],
            "c2": cur["c2"],
            "neighbor_geometric_mean": geom,
            "peak_ratio": ratio,
            "strong_peak": ratio > near_factor,
        })

# Local effective orders between adjacent points are useful to expose the
# oscillatory pattern that a single global p can hide.
local_orders = []
for a, b in zip(rows[:-1], rows[1:]):
    p_local = math.log(b["sigma_E"] / a["sigma_E"]) / math.log(b["dt_ps"] / a["dt_ps"])
    local_orders.append({"dt_lo_ps": a["dt_ps"], "dt_hi_ps": b["dt_ps"], "p_local": p_local})

# Largest contiguous prefix whose C2 values stay within clean_factor of one
# another.  This mirrors the conservative criterion used during smoothing.
prefix_end = rows[0]["dt_ps"]
prefix_len = 1
for n in range(2, len(rows) + 1):
    vals = [x["c2"] for x in rows[:n]]
    if max(vals) / min(vals) <= clean_factor:
        prefix_len = n
        prefix_end = rows[n - 1]["dt_ps"]
    else:
        break

out = {
    "kind": "tel22_ibi_fine_timestep_response_diagnostic",
    "source_report": str(src),
    "median_c2": median_c2,
    "c2_spread": spread,
    "near_factor": near_factor,
    "clean_factor": clean_factor,
    "clean_prefix_points": prefix_len,
    "clean_prefix_max_dt_ps": prefix_end,
    "rows": rows,
    "local_peaks": local_peaks,
    "local_orders": local_orders,
}
dst.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("\n[TEL22_IBI FINE TIMESTEP RESPONSE SUMMARY]")
print(f"[C2] median={median_c2:.6g} min={min_c2:.6g} max={max_c2:.6g} spread={spread:.3f}")
print(f"[PREFIX] contiguous <= {clean_factor:g}x through dt={prefix_end:g} ps ({prefix_len}/{len(rows)} points)")
print("[MAP] dt_ps       sigma_E       sigma_E/dt^2    C2/median   drift       band")
for r in rows:
    print(
        f"      {r['dt_ps']:<10g} {r['sigma_E']:<13.6g} {r['c2']:<15.6g} "
        f"{r['c2_over_median']:<11.3f} {r['relative_block_mean_drift']:<11.3e} {r['band']}"
    )
if local_peaks:
    for p in local_peaks:
        print(
            f"[PEAK] dt={p['dt_ps']:g} ps C2={p['c2']:.6g} "
            f"ratio_vs_neighbor_geom={p['peak_ratio']:.3f} strong={p['strong_peak']}"
        )
else:
    print("[PEAK] none")
amplified = [r for r in rows if r["band"] == "amplified"]
if amplified:
    print("[AMPLIFIED] " + " ".join(f"dt={r['dt_ps']:g}:C2/med={r['c2_over_median']:.3f}" for r in amplified))
else:
    print("[AMPLIFIED] none")
print("[LOCAL ORDER]")
for x in local_orders:
    print(f"  {x['dt_lo_ps']:g} -> {x['dt_hi_ps']:g} ps : p={x['p_local']:.4f}")
print(f"[REPORT] {dst}")
print("[DONE] Diagnostic only. Production IBI priors were not modified.")
PY
