#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi

PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"
DEVICE="${DEVICE:-auto}"
CG_STEPS="${CG_STEPS:-20000}"
CG_DT="${CG_DT:-0.001}"
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH:-link-cell}"
# Traiettoria strutturata per l'analisi.  run_cg_md.py la scrive SOLO se
# --sample_npz e' passato: senza, la produzione gira ma non lascia nulla da
# analizzare, e la g(r), la W1 e il confronto appaiato sulle copie non hanno
# input.  Lo script diagnostico 35 la passava, questo no.
SAMPLE_NPZ="${SAMPLE_NPZ:-samples.npz}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"

cd "${SCRIPT_DIR}"

for path in tel22_model.pt tel22_training_config.json cg_priors.json rigid_bodies_info.json tel22_dataset.bin equilibrated.npz; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
    --model tel22_model.pt \
    --config tel22_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --checkpoint equilibrated.npz \
    --steps "${CG_STEPS}" \
    --dt "${CG_DT}" \
    --kT 2.49 \
    --device "${DEVICE}" \
    --neighbor_search "${NEIGHBOR_SEARCH}" \
    --sample_npz "${SAMPLE_NPZ}" \
    --log_interval "${LOG_INTERVAL}"

echo
echo "[DONE] Produzione completata."
if [ -s "${SAMPLE_NPZ}" ]; then
    echo "[INFO] Traiettoria per l'analisi: ${SAMPLE_NPZ}"
    echo "[INFO] Confronto strutturale col riferimento all-atom:"
    echo "  D=tel22_dataset.bin"
    echo "  python3 diagnostics/scripts/44_tel22_rdf.py \"\$D\" run=${SAMPLE_NPZ}"
    echo "  python3 diagnostics/scripts/42_paired_copy_compare.py \"\$D\" run=${SAMPLE_NPZ}"
else
    echo "[ERROR] ${SAMPLE_NPZ} non prodotto: l'analisi strutturale non e' possibile." >&2
    exit 2
fi
