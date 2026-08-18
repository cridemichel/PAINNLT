#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK_ROOT="$(cd "${TEL22_DIR}/../.." && pwd)"

if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi

PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"
PROFILE_STEPS="${PROFILE_STEPS:-200}"
PROFILE_REPEATS="${PROFILE_REPEATS:-3}"
PROFILE_WARMUP_CALLS="${PROFILE_WARMUP_CALLS:-20}"
PROFILE_DT="${PROFILE_DT:-0.001}"
PROFILE_DEVICE="${PROFILE_DEVICE:-cpu}"
PROFILE_PRECISION="${PROFILE_PRECISION:-float32}"
PROFILE_NEIGHBOR_SEARCH="${PROFILE_NEIGHBOR_SEARCH:-link-cell}"
OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-${TEL22_DIR}/diagnostics/profiling/painn_runtime_baseline}"
MODE="overwrite"

usage() {
    cat <<'USAGE'
Usage: 10_profile_painn_runtime.sh [--dry-run|--overwrite|--resume]

CPU-reference PaiNN runtime profiler. No model/physics parameters are changed.
Environment overrides:
  PROFILE_STEPS=200
  PROFILE_REPEATS=3
  PROFILE_WARMUP_CALLS=20
  PROFILE_DT=0.001
  PROFILE_DEVICE=cpu              (profiling currently requires cpu)
  PROFILE_PRECISION=float32
  PROFILE_NEIGHBOR_SEARCH=link-cell
  PROFILE_OUTPUT_DIR=/path
  PYRESSO=/path/to/pypresso
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) MODE="dry-run" ;;
        --overwrite) MODE="overwrite" ;;
        --resume) MODE="resume" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

case "${PROFILE_STEPS}" in ''|*[!0-9]*) echo "[ERROR] PROFILE_STEPS must be a positive integer" >&2; exit 2;; esac
case "${PROFILE_REPEATS}" in ''|*[!0-9]*) echo "[ERROR] PROFILE_REPEATS must be a positive integer" >&2; exit 2;; esac
case "${PROFILE_WARMUP_CALLS}" in ''|*[!0-9]*) echo "[ERROR] PROFILE_WARMUP_CALLS must be a non-negative integer" >&2; exit 2;; esac
if [[ "${PROFILE_STEPS}" -le 0 || "${PROFILE_REPEATS}" -le 0 ]]; then
    echo "[ERROR] PROFILE_STEPS and PROFILE_REPEATS must be > 0" >&2
    exit 2
fi
if [[ "${PROFILE_DEVICE}" != "cpu" ]]; then
    echo "[ERROR] This profiler is intentionally CPU-reference only; set PROFILE_DEVICE=cpu." >&2
    exit 2
fi

for path in \
    "${TEL22_DIR}/tel22_model.pt" \
    "${TEL22_DIR}/tel22_training_config.json" \
    "${TEL22_DIR}/cg_priors.json" \
    "${TEL22_DIR}/rigid_bodies_info.json" \
    "${TEL22_DIR}/tel22_dataset.bin" \
    "${TEL22_DIR}/equilibrated.npz"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing required TEL22 input: ${path}" >&2
        exit 1
    fi
done

if [[ "${MODE}" == "overwrite" ]]; then
    rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

BASELINE_COMMIT="unknown"
CURRENT_COMMIT="unknown"
if command -v git >/dev/null 2>&1 && git -C "${FRAMEWORK_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    CURRENT_COMMIT="$(git -C "${FRAMEWORK_ROOT}" rev-parse HEAD)"
    if git -C "${FRAMEWORK_ROOT}" rev-parse painn-runtime-baseline^{commit} >/dev/null 2>&1; then
        BASELINE_COMMIT="$(git -C "${FRAMEWORK_ROOT}" rev-parse painn-runtime-baseline^{commit})"
    fi
fi

cat <<EOF2
[TEL22 PAINN RUNTIME PROFILE -- CPU REFERENCE]
baseline tag/commit : ${BASELINE_COMMIT}
current commit      : ${CURRENT_COMMIT}
model               : tel22_model.pt
device / precision  : ${PROFILE_DEVICE} / ${PROFILE_PRECISION}
neighbor search     : ${PROFILE_NEIGHBOR_SEARCH}
dt                   : ${PROFILE_DT} ps
steps / repeat      : ${PROFILE_STEPS}
repeats             : ${PROFILE_REPEATS}
warmup force calls  : ${PROFILE_WARMUP_CALLS}
logging             : disabled; one integrator.run per repeat
output              : ${OUTPUT_DIR}
[NOTE] Profiling is opt-in and does not change architecture, weights, cutoff, RBF, priors, or precision.
[NOTE] host_payload_lower_bound excludes allocator/map-node overhead and libtorch internal allocations.
EOF2

if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY-RUN] Would execute ${PROFILE_REPEATS} identical NVE repeats from equilibrated.npz."
    echo "[DRY-RUN] Each repeat invokes run_cg_md.py with --painn_profile_report and CPU FP32."
    echo "[DRY-RUN] No MD executed."
    exit 0
fi

repeat=1
while [[ "${repeat}" -le "${PROFILE_REPEATS}" ]]; do
    report="${OUTPUT_DIR}/repeat_${repeat}.json"
    log="${OUTPUT_DIR}/repeat_${repeat}.log"
    if [[ "${MODE}" == "resume" && -s "${report}" ]]; then
        echo "[REUSE] repeat ${repeat}: ${report}"
        repeat=$((repeat + 1))
        continue
    fi
    echo "[RUN] repeat ${repeat}/${PROFILE_REPEATS}"
    (
        cd "${TEL22_DIR}"
        "${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
            --model tel22_model.pt \
            --config tel22_training_config.json \
            --priors cg_priors.json \
            --rb_info rigid_bodies_info.json \
            --dataset tel22_dataset.bin \
            --checkpoint equilibrated.npz \
            --steps "${PROFILE_STEPS}" \
            --dt "${PROFILE_DT}" \
            --kT 2.49 \
            --device "${PROFILE_DEVICE}" \
            --ml_precision "${PROFILE_PRECISION}" \
            --neighbor_search "${PROFILE_NEIGHBOR_SEARCH}" \
            --nve \
            --no_log \
            --no_vtf \
            --log_interval "${PROFILE_STEPS}" \
            --painn_profile_warmup_calls "${PROFILE_WARMUP_CALLS}" \
            --painn_profile_report "${report}"
    ) 2>&1 | tee "${log}"
    if [[ ! -s "${report}" ]]; then
        echo "[ERROR] Profiling report was not produced: ${report}" >&2
        echo "        Rebuild the patched ESPResSo plugin before rerunning:" >&2
        echo "        bash simulation/espresso_plugin/copy_plugin_files.sh" >&2
        exit 1
    fi
    repeat=$((repeat + 1))
done

python3 - "${OUTPUT_DIR}" "${PROFILE_REPEATS}" "${BASELINE_COMMIT}" "${CURRENT_COMMIT}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
repeats = int(sys.argv[2])
baseline_commit = sys.argv[3]
current_commit = sys.argv[4]
reports = []
for idx in range(1, repeats + 1):
    path = out / f"repeat_{idx}.json"
    with path.open() as handle:
        reports.append(json.load(handle))

stage_keys = [
    "node_index_mean",
    "neighbor_traversal_mean",
    "edge_pack_mean",
    "tensor_inputs_mean",
    "forward_mean",
    "energy_scalar_mean",
    "autograd_mean",
    "force_to_cpu_mean",
    "force_scatter_mean",
    "unattributed_cleanup_mean",
]

def values(fn):
    return [float(fn(item)) for item in reports]

def stats(vals):
    mean = statistics.fmean(vals)
    return {
        "mean": mean,
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "pstdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "cv": (statistics.pstdev(vals) / mean if len(vals) > 1 and mean else 0.0),
    }

wall = stats(values(lambda x: x["integration"]["wall_ms_per_step"]))
painn = stats(values(lambda x: x["timings_ms"]["total_mean"]))
stages = {key.removesuffix("_mean"): stats(values(lambda x, k=key: x["timings_ms"][k])) for key in stage_keys}
median_total = painn["median"]
for entry in stages.values():
    entry["fraction_of_median_painn"] = entry["median"] / median_total if median_total else 0.0

ranked = sorted(stages.items(), key=lambda kv: kv[1]["median"], reverse=True)
graph = {
    "particles_mean_median": statistics.median(values(lambda x: x["graph"]["particles_mean"])),
    "directed_edges_mean_median": statistics.median(values(lambda x: x["graph"]["directed_edges_mean"])),
    "physical_pairs_mean_median": statistics.median(values(lambda x: x["graph"]["physical_pairs_mean"])),
}
allocation = {
    "host_payload_lower_bound_bytes_mean_median": statistics.median(values(lambda x: x["allocation_churn_indicators"]["host_payload_lower_bound_bytes_mean"])),
    "temporary_cpp_containers_per_call": int(reports[0]["allocation_churn_indicators"]["temporary_cpp_containers_per_call"]),
    "note": reports[0]["allocation_churn_indicators"]["note"],
}
summary = {
    "schema_version": 1,
    "kind": "tel22_painn_runtime_profile_cpu_reference",
    "baseline_commit": baseline_commit,
    "current_commit": current_commit,
    "repeats": repeats,
    "runtime_config": reports[0]["runtime_config"],
    "wall_ms_per_step": wall,
    "painn_total_ms_per_force_call": painn,
    "stages_ms_per_force_call": stages,
    "hotspot_ranking": [name for name, _ in ranked],
    "graph": graph,
    "allocation_churn_indicators": allocation,
    "raw_reports": [f"repeat_{idx}.json" for idx in range(1, repeats + 1)],
}
summary_path = out / "painn_runtime_profile_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

print("\n[TEL22 PAINN PROFILE SUMMARY]")
print(f"[WALL] median={wall['median']:.6g} ms/step mean={wall['mean']:.6g} CV={wall['cv']:.3%}")
print(f"[PAINN] median={painn['median']:.6g} ms/force-call mean={painn['mean']:.6g} CV={painn['cv']:.3%}")
print(
    "[GRAPH] "
    f"N={graph['particles_mean_median']:.1f} "
    f"directed_E={graph['directed_edges_mean_median']:.1f} "
    f"pairs={graph['physical_pairs_mean_median']:.1f}"
)
print(
    "[ALLOC-INDICATOR] "
    f"host_payload_lower_bound={allocation['host_payload_lower_bound_bytes_mean_median'] / 1024.0:.3f} KiB/call "
    f"temporary_cpp_containers={allocation['temporary_cpp_containers_per_call']}/call"
)
print("[STAGES] median ms/force-call and share of PaiNN total")
for name, entry in ranked:
    print(f"  {name:20s} {entry['median']:10.6f} ms  {entry['fraction_of_median_painn']:8.2%}")
print("[HOTSPOT] " + " > ".join(name for name, _ in ranked[:5]))
print(f"[REPORT] {summary_path}")
PY
