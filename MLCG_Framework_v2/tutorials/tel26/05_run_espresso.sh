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
# MODEL scegli quale checkpoint simulare.  tel26_model.pt e' quello salvato
#       dall'early stopping sulla validation loss, cioe' il criterio che sui
#       quattro modelli di riferimento ordina AL CONTRARIO: per lo sweep si
#       passano i checkpoint periodici, tel26_model.ep*.pt.
# DISABLE_ML=1  tiene il modello per provenienza ma non attiva PaiNN.  E' il
#       controllo solo-prior: stesso stato iniziale, stesse interazioni
#       classiche, nessun residuo ML.  Serve sia a distinguere un problema di
#       modello da uno di impostazione, sia come terza curva del confronto
#       sulla g(r), dove senza di essa un accordo non e' attribuibile.
MODEL="${MODEL:-tel26_model.pt}"
# Deve combaciare con quello scritto da 04: vedi li' il perche'.
CHECKPOINT="${CHECKPOINT:-equilibrated.npz}"
ml_args=()
[ -n "${DISABLE_ML:-}" ] && ml_args+=(--disable_ml)

cd "${SCRIPT_DIR}"

for path in "${MODEL}" tel26_training_config.json cg_priors.json rigid_bodies_info.json tel26_dataset.bin "${CHECKPOINT}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
    --model "${MODEL}" \
    ${ml_args[@]+"${ml_args[@]}"} \
    --config tel26_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel26_dataset.bin \
    --checkpoint "${CHECKPOINT}" \
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
    echo "[INFO] Confronto strutturale col riferimento all-atom mappato:"
    echo "  python3 ../tel22/diagnostics/scripts/44_tel22_rdf.py \\"
    echo "      tel26_dataset.bin run=${SAMPLE_NPZ} --nuc 26"
    echo "       Il canale B3-B3 porta i legami di Hoogsteen: e' li' che si vede"
    echo "       se le tetradi reggono.  Serve la terza curva per attribuire un"
    echo "       accordo al residuo ML invece che ai prior -- rilancia con"
    echo "       CLASSICAL=1 DISABLE_ML=1 SAMPLE_NPZ=samples_priors.npz."
    echo
    echo "[NOTA] Restano indisponibili per TEL26 le coordinate collettive dello"
    echo "       stadio select (Q sul ciclo delle tetradi, RMSD): passano da"
    echo "       _hb_common e vogliono il registro delle tetradi del sistema,"
    echo "       non il solo numero di residui."
else
    echo "[ERROR] ${SAMPLE_NPZ} non prodotto: l'analisi strutturale non e' possibile." >&2
    exit 2
fi
