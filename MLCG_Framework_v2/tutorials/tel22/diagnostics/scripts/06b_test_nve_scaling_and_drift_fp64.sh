#!/usr/bin/env bash
set -euo pipefail

# TEL22 FP64 A/B diagnostic for NVE timestep scaling and energy drift.
# Standalone by design: it does not depend on another diagnostic wrapper.
# The protocol matches the accepted FP32 recheck and changes only ML precision.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/tel22_training_config.json" ]]; then
    TUTORIAL_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../../tel22_training_config.json" ]]; then
    TUTORIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    echo "[ERROR] Cannot locate tutorials/tel22 from ${SCRIPT_DIR}" >&2
    exit 1
fi
FRAMEWORK_ROOT="$(cd "${TUTORIAL_DIR}/../.." && pwd)"
CERTIFIER="${FRAMEWORK_ROOT}/simulation/certify_nve.py"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"

# Frozen A/B protocol: identical to the accepted TEL22 FP32 recheck except
# for ML_PRECISION=float64.  Keep the output separate from the FP32 baseline.
ML_DEVICE="cpu"
ML_PRECISION="float64"
NEIGHBOR_SEARCH="link-cell"
DTS="0.001 0.0015 0.002 0.003 0.004 0.005"
DURATION_PS="2.0"
OUTPUT_DIR="diagnostics/nve/nve_scaling_drift_recheck_fp64"
SLOPE_MIN="1.7"
SLOPE_MAX="2.3"
MIN_R2="0.97"
MAX_RELATIVE_DRIFT="1e-4"

usage() {
    cat <<'USAGE'
Usage:
  06b_test_nve_scaling_and_drift_fp64.sh [--dry-run | --overwrite | --resume]

This is a frozen A/B diagnostic.  It matches the accepted TEL22 FP32 recheck
and changes only PaiNN inference precision from float32 to float64.
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

cd "${TUTORIAL_DIR}"
for path in \
    tel22_model.pt \
    tel22_model.pt.manifest.json \
    tel22_training_config.json \
    cg_priors.json \
    rigid_bodies_info.json \
    tel22_dataset.bin \
    equilibrated.npz; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing required input: ${path}" >&2; exit 1; }
done
[[ -f "${CERTIFIER}" ]] || { echo "[ERROR] Missing certifier: ${CERTIFIER}" >&2; exit 1; }

read -r -a DT_ARGS <<< "${DTS}"
((${#DT_ARGS[@]} >= 3)) || { echo "[ERROR] DTS needs at least three values" >&2; exit 1; }

cat <<EOF_PLAN
[TEL22 NVE FP64 A/B RECHECK]
Reference        : accepted TEL22 FP32 recheck
Changed variable : PaiNN inference precision float32 -> float64
Hamiltonian      : cg_priors.json + tel22_model.pt (PaiNN active)
checkpoint       : equilibrated.npz
dt grid [ps]     : ${DTS}
duration / dt    : ${DURATION_PS} ps
sampling         : every integration step
thermostat       : OFF (NVE)
device           : ${ML_DEVICE}
precision        : ${ML_PRECISION}
neighbor search  : ${NEIGHBOR_SEARCH}
scaling gate     : ${SLOPE_MIN} <= p <= ${SLOPE_MAX}; R2 >= ${MIN_R2}
drift gate       : relative block-mean drift <= ${MAX_RELATIVE_DRIFT}
output           : ${OUTPUT_DIR}
EOF_PLAN

CERT_CMD=(
    python3 "${CERTIFIER}"
    --pypresso "${PYPRESSO}"
    --model tel22_model.pt
    --config tel22_training_config.json
    --priors cg_priors.json
    --rb-info rigid_bodies_info.json
    --dataset tel22_dataset.bin
    --checkpoint equilibrated.npz
    --dts "${DT_ARGS[@]}"
    --duration-ps "${DURATION_PS}"
    --device "${ML_DEVICE}"
    --ml-precision "${ML_PRECISION}"
    --neighbor-search "${NEIGHBOR_SEARCH}"
    --output-dir "${OUTPUT_DIR}"
    --slope-min "${SLOPE_MIN}"
    --slope-max "${SLOPE_MAX}"
    --min-r2 "${MIN_R2}"
    --max-relative-drift "${MAX_RELATIVE_DRIFT}"
)
case "${MODE}" in
    dry-run) CERT_CMD+=(--dry-run) ;;
    overwrite) CERT_CMD+=(--overwrite) ;;
    resume) CERT_CMD+=(--reuse-existing) ;;
esac
"${CERT_CMD[@]}"

[[ "${MODE}" == "dry-run" ]] && exit 0
REPORT="${OUTPUT_DIR}/nve_certification_report.json"
[[ -f "${REPORT}" ]] || { echo "[ERROR] Missing report: ${REPORT}" >&2; exit 1; }

python3 - "${REPORT}" <<'PY'
import json, math, sys
p = sys.argv[1]
r = json.load(open(p, encoding="utf-8"))
c = r["certification"]
s = c["scaling"]
runs = sorted(r["runs"], key=lambda x: float(x["dt_ps"]))
print("\n[TEL22 NVE FP64 A/B SUMMARY]")
print(f"[SCALING] p={float(s['exponent_p']):.6f} R2={float(s['loglog_r2']):.6f} pass={c['scaling_pass']}")
worst = max(runs, key=lambda x: float(x["relative_block_mean_drift"]))
print(f"[DRIFT] max={float(worst['relative_block_mean_drift']):.3e} at dt={float(worst['dt_ps']):g} ps pass={c['drift_pass']}")
print("[TABLE] dt_ps       sigma_E       sigma_E/dt^2    rel_block_drift")
c2 = []
for x in runs:
    dt = float(x["dt_ps"]); sig = float(x["sigma_E"]); drift = float(x["relative_block_mean_drift"])
    val = sig / (dt * dt); c2.append(val)
    print(f"        {dt:<10g} {sig:<13.6g} {val:<15.6g} {drift:.3e}")
pos = [x for x in c2 if x > 0 and math.isfinite(x)]
if pos:
    print(f"[C2] max/min spread={max(pos)/min(pos):.3f} (diagnostic)")
print(f"[FINAL] pass={c['pass']}")
print(f"[REPORT] {p}")
PY
