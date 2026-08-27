#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROBE="${SCRIPT_DIR}/25_test_mps_memory_growth.sh"
SUMMARIZER="${SCRIPT_DIR}/26_summarize_mps_empty_cache_ab.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_REL="${MPS_EMPTYCACHE_AB_OUT:-diagnostics/memory/tel22_uniform_a0p255_smoke3_mps_emptycache_ab_5k}"
if [[ "${OUT_REL}" = /* ]]; then OUT="${OUT_REL}"; else OUT="${TEL22_DIR}/${OUT_REL}"; fi
BASE_OUT="${OUT}/baseline_cache_default"
CADENCE="${MPS_EMPTYCACHE_CADENCE:-100}"
CAND_OUT="${OUT}/emptycache_e${CADENCE}"
STEPS="${MPS_EMPTYCACHE_AB_STEPS:-5000}"
ABORT_MIB="${MPS_MEMORY_ABORT_MIB:-50000}"
SUMMARY="${OUT}/mps_empty_cache_ab_summary.json"

usage() {
cat <<'USAGE'
Usage:
  26_test_mps_empty_cache_ab.sh [--dry-run | --overwrite | --resume]

Controlled TEL22/MPS allocator A/B from the same checkpoint:
  A: MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS=0
  B: MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS=100 (override supported)

The bridge default remains 0/OFF. The candidate calls emptyCache only after a
successful force evaluation has returned and all per-call tensors are dead.

Useful overrides:
  MPS_EMPTYCACHE_AB_STEPS=5000
  MPS_EMPTYCACHE_CADENCE=100
  MPS_MEMORY_ABORT_MIB=50000
  MPS_EMPTYCACHE_AB_OUT=diagnostics/memory/my_emptycache_ab
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
(($# == 0)) || { echo "[ERROR] Unexpected arguments: $*" >&2; exit 2; }
for pair in "MPS_EMPTYCACHE_AB_STEPS:${STEPS}" "MPS_EMPTYCACHE_CADENCE:${CADENCE}"; do
    name="${pair%%:*}"; value="${pair#*:}"
    case "${value}" in ''|*[!0-9]*) echo "[ERROR] ${name} must be a positive integer" >&2; exit 2;; esac
    [[ "${value}" -gt 0 ]] || { echo "[ERROR] ${name} must be > 0" >&2; exit 2; }
done
for path in "${PROBE}" "${SUMMARIZER}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] Missing component: ${path}" >&2; exit 1; }
done

case "${MODE}" in
    dry-run) CHILD_MODE="--dry-run" ;;
    overwrite) CHILD_MODE="--overwrite" ;;
    resume) CHILD_MODE="--resume" ;;
    normal) CHILD_MODE="" ;;
esac

run_variant() {
    local label="$1" cadence="$2" output="$3"
    echo "[A/B ${label}] emptyCache cadence=${cadence} successful force calls"
    env \
      MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS="${cadence}" \
      MPS_MEMORY_OUT="${output}" \
      MPS_MEMORY_STEPS="${STEPS}" \
      MPS_MEMORY_WARMUP_STEP=2000 \
      MPS_MEMORY_FOOTPRINT_EVERY_SAMPLES=5 \
      MPS_MEMORY_ABORT_MIB="${ABORT_MIB}" \
      bash "${PROBE}" ${CHILD_MODE:+"${CHILD_MODE}"}
}

cat <<EOF
[TEL22 MPS EMPTYCACHE A/B]
steps              : ${STEPS}
baseline cadence   : 0 (disabled)
candidate cadence  : ${CADENCE}
memory guard       : ${ABORT_MIB} MiB
output             : ${OUT_REL}
EOF

run_variant BASELINE 0 "${BASE_OUT}"
run_variant CANDIDATE "${CADENCE}" "${CAND_OUT}"
if [[ "${MODE}" == "dry-run" ]]; then exit 0; fi

"${PYTHON_BIN}" "${SUMMARIZER}" \
  --baseline "${BASE_OUT}/mps_memory_summary.json" \
  --candidate "${CAND_OUT}/mps_memory_summary.json" \
  --baseline-log "${BASE_OUT}/run.log" \
  --candidate-log "${CAND_OUT}/run.log" \
  --candidate-cadence "${CADENCE}" \
  --output "${SUMMARY}"
echo "[DONE] ${SUMMARY}"
