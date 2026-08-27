#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TEL22_DIR}/../.." && pwd)"
MONITOR="${SCRIPT_DIR}/25_monitor_mps_memory.py"
RUNNER="${FRAMEWORK_ROOT}/simulation/run_cg_md.py"
BRIDGE_SOURCE="${FRAMEWORK_ROOT}/simulation/espresso_plugin/PaiNN_ML_Potential.cpp"
INSTALLED_BRIDGE_SOURCE="${FRAMEWORK_ROOT}/espresso/src/core/nonbonded_interactions/PaiNN_ML_Potential.cpp"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYPRESSO="${PYPRESSO:-${DEFAULT_PYPRESSO}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CANDIDATE_REL="${MPS_MEMORY_CANDIDATE:-diagnostics/full_pipeline/tel22_uniform_morse_a0p255_smoke3}"
if [[ "${CANDIDATE_REL}" = /* ]]; then
    CANDIDATE="${CANDIDATE_REL}"
else
    CANDIDATE="${TEL22_DIR}/${CANDIDATE_REL}"
fi
OUT_REL="${MPS_MEMORY_OUT:-diagnostics/memory/tel22_uniform_a0p255_smoke3_mps_5k}"
if [[ "${OUT_REL}" = /* ]]; then
    OUT="${OUT_REL}"
else
    OUT="${TEL22_DIR}/${OUT_REL}"
fi

STEPS="${MPS_MEMORY_STEPS:-5000}"
DT="${MPS_MEMORY_DT:-0.001}"
LOG_INTERVAL="${MPS_MEMORY_LOG_INTERVAL:-10}"
SAMPLE_SECONDS="${MPS_MEMORY_SAMPLE_SECONDS:-2}"
PRINT_EVERY="${MPS_MEMORY_PRINT_EVERY_SAMPLES:-15}"
FOOTPRINT_EVERY="${MPS_MEMORY_FOOTPRINT_EVERY_SAMPLES:-5}"
WARMUP_STEP="${MPS_MEMORY_WARMUP_STEP:-2000}"
GROWTH_THRESHOLD_MIB="${MPS_MEMORY_GROWTH_THRESHOLD_MIB:-1024}"
SLOPE_THRESHOLD="${MPS_MEMORY_SLOPE_THRESHOLD_MIB_PER_1000_STEPS:-256}"
ABORT_MEMORY_MIB="${MPS_MEMORY_ABORT_MIB:-0}"
NEIGHBOR_SEARCH="${MPS_MEMORY_NEIGHBOR_SEARCH:-link-cell}"
KT="${MPS_MEMORY_KT:-2.49}"

MODEL="${CANDIDATE}/artifacts/tel22_model.pt"
CONFIG="${CANDIDATE}/artifacts/tel22_training_config.json"
PRIORS="${CANDIDATE}/inputs/cg_priors.json"
RB_INFO="${CANDIDATE}/artifacts/rigid_bodies_info.json"
DATASET="${CANDIDATE}/artifacts/tel22_dataset.bin"
CHECKPOINT="${CANDIDATE}/artifacts/equilibrated.npz"
MODEL_MANIFEST="${MODEL}.manifest.json"
ENERGY="${OUT}/energy.csv"
FINAL_STATE="${OUT}/final_state.npz"
SUMMARY="${OUT}/mps_memory_summary.json"

usage() {
cat <<'USAGE'
Usage:
  25_test_mps_memory_growth.sh [--dry-run | --overwrite | --resume]

External TEL22 MPS process-memory diagnostic. It runs an isolated NVT from the
completed test-24 candidate and samples process-tree RSS, macOS physical
footprint (when vmmap is available), swap, and the flushed integration step.

No model, prior, bridge file, or production file is modified. MPS defaults to
emptyCache every 100 successful force calls; an environment override is
recorded in the report and the effective policy is attested in run.log.
This test diagnoses memory growth; it does not claim that growth is a leak.

Useful overrides:
  MPS_MEMORY_STEPS=5000
  MPS_MEMORY_SAMPLE_SECONDS=2
  MPS_MEMORY_WARMUP_STEP=2000
  MPS_MEMORY_FOOTPRINT_EVERY_SAMPLES=5
  MPS_MEMORY_ABORT_MIB=0           # RSS/physical-footprint guard; 0 disables it
  MPS_MEMORY_CANDIDATE=diagnostics/full_pipeline/tel22_uniform_morse_a0p255_smoke3
  MPS_MEMORY_OUT=diagnostics/memory/my_probe
  PYPRESSO=/path/to/pypresso
USAGE
}

MODE="normal"
case "${1:-}" in
    "") ;;
    --dry-run) MODE="dry-run"; shift ;;
    --overwrite) MODE="overwrite"; shift ;;
    --resume) MODE="resume"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac
if (($# != 0)); then
    echo "[ERROR] Unexpected arguments: $*" >&2
    exit 2
fi

positive_integer() {
    local name="$1" value="$2"
    case "${value}" in
        ''|*[!0-9]*) echo "[ERROR] ${name} must be a positive integer" >&2; exit 2 ;;
    esac
    if [[ "${value}" -le 0 ]]; then
        echo "[ERROR] ${name} must be > 0" >&2
        exit 2
    fi
}
nonnegative_integer() {
    local name="$1" value="$2"
    case "${value}" in
        ''|*[!0-9]*) echo "[ERROR] ${name} must be a non-negative integer" >&2; exit 2 ;;
    esac
}
positive_integer MPS_MEMORY_STEPS "${STEPS}"
positive_integer MPS_MEMORY_LOG_INTERVAL "${LOG_INTERVAL}"
positive_integer MPS_MEMORY_PRINT_EVERY_SAMPLES "${PRINT_EVERY}"
positive_integer MPS_MEMORY_FOOTPRINT_EVERY_SAMPLES "${FOOTPRINT_EVERY}"
nonnegative_integer MPS_MEMORY_WARMUP_STEP "${WARMUP_STEP}"

if [[ "${MODE}" != "dry-run" && "$(uname -s)" != "Darwin" ]]; then
    echo "[ERROR] The MPS memory diagnostic requires macOS." >&2
    exit 1
fi
for path in "${MONITOR}" "${RUNNER}" "${BRIDGE_SOURCE}" "${INSTALLED_BRIDGE_SOURCE}" \
            "${MODEL}" "${MODEL_MANIFEST}" "${CONFIG}" \
            "${PRIORS}" "${RB_INFO}" "${DATASET}" "${CHECKPOINT}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing required input: ${path}" >&2; exit 1; }
done
if ! cmp -s "${BRIDGE_SOURCE}" "${INSTALLED_BRIDGE_SOURCE}"; then
    echo "[ERROR] ESPResSo contains a stale PaiNN bridge source." >&2
    echo "        Run: bash simulation/espresso_plugin/copy_plugin_files.sh" >&2
    echo "        Then rebuild ESPResSo before repeating this diagnostic." >&2
    exit 1
fi
if [[ "${MODE}" != "dry-run" ]]; then
    command -v "${PYPRESSO}" >/dev/null 2>&1 || [[ -x "${PYPRESSO}" ]] || {
        echo "[ERROR] pypresso not found: ${PYPRESSO}" >&2
        exit 1
    }
fi

monitor_cmd=(
    "${PYTHON_BIN}" "${MONITOR}"
    --output-dir "${OUT}"
    --energy-file "${ENERGY}"
    --expected-final-step "${STEPS}"
    --sample-interval-seconds "${SAMPLE_SECONDS}"
    --print-every-samples "${PRINT_EVERY}"
    --footprint-every-samples "${FOOTPRINT_EVERY}"
    --warmup-step "${WARMUP_STEP}"
    --growth-threshold-mib "${GROWTH_THRESHOLD_MIB}"
    --slope-threshold-mib-per-1000-steps "${SLOPE_THRESHOLD}"
    --abort-memory-mib "${ABORT_MEMORY_MIB}"
    --input "model=${MODEL}"
    --input "model_manifest=${MODEL_MANIFEST}"
    --input "config=${CONFIG}"
    --input "priors=${PRIORS}"
    --input "rb_info=${RB_INFO}"
    --input "dataset=${DATASET}"
    --input "checkpoint=${CHECKPOINT}"
    --input "bridge_source=${BRIDGE_SOURCE}"
    --input "installed_bridge_source=${INSTALLED_BRIDGE_SOURCE}"
    --
    "${PYPRESSO}" "${RUNNER}"
    --model "${MODEL}"
    --config "${CONFIG}"
    --priors "${PRIORS}"
    --rb_info "${RB_INFO}"
    --dataset "${DATASET}"
    --checkpoint "${CHECKPOINT}"
    --steps "${STEPS}"
    --dt "${DT}"
    --kT "${KT}"
    --device mps
    --ml_precision float32
    --neighbor_search "${NEIGHBOR_SEARCH}"
    --energy_file "${ENERGY}"
    --out_checkpoint "${FINAL_STATE}"
    --log_interval "${LOG_INTERVAL}"
    --no_vtf
)

cat <<EOF
[TEL22 PAINN MPS MEMORY DIAGNOSTIC]
candidate         : ${CANDIDATE_REL}
device/precision  : mps / float32
ensemble          : NVT, kT=${KT}
steps / dt        : ${STEPS} / ${DT} ps
sampling interval : ${SAMPLE_SECONDS} s
warmup step       : ${WARMUP_STEP}
memory safety guard: ${ABORT_MEMORY_MIB} MiB (0=disabled)
MPS emptyCache cadence: ${MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS:-100 (bridge default)}
output            : ${OUT_REL}
[NOTE] This run is isolated and never overwrites test-24 or production artifacts.
[NOTE] Memory classification is diagnostic and cannot alone distinguish live tensors from allocator cache.
EOF

if [[ "${MODE}" == "dry-run" ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "${monitor_cmd[@]}"
    printf '\n'
    exit 0
fi

if [[ "${MODE}" == "overwrite" ]]; then
    if [[ "${OUT}" == "/" || "${OUT}" == "${TEL22_DIR}" || -z "${OUT}" ]]; then
        echo "[ERROR] Refusing unsafe output removal: ${OUT}" >&2
        exit 1
    fi
    rm -rf "${OUT}"
elif [[ "${MODE}" == "resume" ]]; then
    if [[ -s "${SUMMARY}" ]] && "${PYTHON_BIN}" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["run"]["complete"] else 1)' \
        "${SUMMARY}"; then
        echo "[REUSE] completed diagnostic: ${SUMMARY}"
        exit 0
    fi
    if [[ -e "${OUT}" ]]; then
        echo "[ERROR] An incomplete memory trajectory cannot be resumed." >&2
        echo "        Use --overwrite to start a fresh diagnostic from the same checkpoint." >&2
        exit 1
    fi
elif [[ -e "${OUT}" ]]; then
    echo "[ERROR] Output already exists: ${OUT}" >&2
    echo "        Use --overwrite or --resume." >&2
    exit 1
fi

mkdir -p "${OUT}"
"${monitor_cmd[@]}"
[[ -s "${SUMMARY}" ]] || { echo "[ERROR] Missing memory summary: ${SUMMARY}" >&2; exit 1; }
echo "[DONE] ${SUMMARY}"
