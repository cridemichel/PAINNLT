#!/usr/bin/env bash
set -euo pipefail

# TEL22: repeat the NVE timestep-scaling and energy-drift check from the
# freshly generated pipeline checkpoint. Diagnostic/certification only.
# Intended destination:
#   tutorials/tel22/diagnostics/scripts/06_test_nve_scaling_and_drift.sh

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
NVE_DEVICE="${NVE_DEVICE:-cpu}"
NVE_ML_PRECISION="${NVE_ML_PRECISION:-float32}"
NVE_NEIGHBOR_SEARCH="${NVE_NEIGHBOR_SEARCH:-link-cell}"
NVE_DTS="${NVE_DTS:-0.001 0.0015 0.002 0.003 0.004 0.005}"
NVE_DURATION_PS="${NVE_DURATION_PS:-2.0}"
NVE_OUTPUT_DIR="${NVE_OUTPUT_DIR:-diagnostics/nve/nve_scaling_drift_recheck}"
NVE_SLOPE_MIN="${NVE_SLOPE_MIN:-1.7}"
NVE_SLOPE_MAX="${NVE_SLOPE_MAX:-2.3}"
NVE_MIN_R2="${NVE_MIN_R2:-0.97}"
NVE_MAX_RELATIVE_DRIFT="${NVE_MAX_RELATIVE_DRIFT:-1e-4}"

usage() {
    cat <<'USAGE'
Usage:
  06_test_nve_scaling_and_drift.sh [--dry-run | --overwrite | --resume]

Environment overrides:
  NVE_DURATION_PS=5.0
  NVE_DTS="0.001 0.0015 0.002 0.003 0.004 0.005"
  NVE_OUTPUT_DIR=diagnostics/nve/my_recheck
  NVE_SLOPE_MIN=1.7
  NVE_SLOPE_MAX=2.3
  NVE_MIN_R2=0.97
  NVE_MAX_RELATIVE_DRIFT=1e-4
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

read -r -a DT_ARGS <<< "${NVE_DTS}"
((${#DT_ARGS[@]} >= 3)) || { echo "[ERROR] NVE_DTS needs at least three values" >&2; exit 1; }

cat <<EOF_PLAN
[TEL22 NVE SCALING + DRIFT RECHECK]
Hamiltonian      : cg_priors.json + tel22_model.pt (PaiNN active)
checkpoint       : equilibrated.npz
dt grid [ps]     : ${NVE_DTS}
duration / dt    : ${NVE_DURATION_PS} ps
sampling         : every integration step
thermostat       : OFF (NVE)
device           : ${NVE_DEVICE}
precision        : ${NVE_ML_PRECISION}
neighbor search  : ${NVE_NEIGHBOR_SEARCH}
scaling gate     : ${NVE_SLOPE_MIN} <= p <= ${NVE_SLOPE_MAX}; R2 >= ${NVE_MIN_R2}
drift gate       : relative block-mean drift <= ${NVE_MAX_RELATIVE_DRIFT}
output           : ${NVE_OUTPUT_DIR}
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
    --duration-ps "${NVE_DURATION_PS}"
    --device "${NVE_DEVICE}"
    --ml-precision "${NVE_ML_PRECISION}"
    --neighbor-search "${NVE_NEIGHBOR_SEARCH}"
    --output-dir "${NVE_OUTPUT_DIR}"
    --slope-min "${NVE_SLOPE_MIN}"
    --slope-max "${NVE_SLOPE_MAX}"
    --min-r2 "${NVE_MIN_R2}"
    --max-relative-drift "${NVE_MAX_RELATIVE_DRIFT}"
)
case "${MODE}" in
    dry-run) CERT_CMD+=(--dry-run) ;;
    overwrite) CERT_CMD+=(--overwrite) ;;
    resume) CERT_CMD+=(--reuse-existing) ;;
esac
"${CERT_CMD[@]}"

[[ "${MODE}" == "dry-run" ]] && exit 0
REPORT="${NVE_OUTPUT_DIR}/nve_certification_report.json"
[[ -f "${REPORT}" ]] || { echo "[ERROR] Missing report: ${REPORT}" >&2; exit 1; }

python3 - "${REPORT}" <<'PY'
import json, math, sys
p = sys.argv[1]
r = json.load(open(p, encoding="utf-8"))
c = r["certification"]
s = c["scaling"]
runs = sorted(r["runs"], key=lambda x: float(x["dt_ps"]))
print("\n[TEL22 NVE RECHECK SUMMARY]")
print(f"[SCALING] p={float(s['exponent_p']):.6f} R2={float(s['loglog_r2']):.6f} pass={c['scaling_pass']}")
worst = max(runs, key=lambda x: float(x["relative_block_mean_drift"]))
print(f"[DRIFT] max={float(worst['relative_block_mean_drift']):.3e} at dt={float(worst['dt_ps']):g} ps pass={c['drift_pass']}")
print("[TABLE] dt_ps       sigma_E       sigma_E/dt^2    rel_block_drift")
c2=[]
for x in runs:
    dt=float(x["dt_ps"]); sig=float(x["sigma_E"]); drift=float(x["relative_block_mean_drift"])
    val=sig/(dt*dt); c2.append(val)
    print(f"        {dt:<10g} {sig:<13.6g} {val:<15.6g} {drift:.3e}")
pos=[x for x in c2 if x>0 and math.isfinite(x)]
if pos: print(f"[C2] max/min spread={max(pos)/min(pos):.3f} (diagnostic)")
print(f"[FINAL] pass={c['pass']}")
print(f"[REPORT] {p}")
PY
