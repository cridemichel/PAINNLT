#!/usr/bin/env bash
# Produzione CG MD con campionamento strutturato, per il confronto
# termodinamico: riferimento / solo-prior / prior+rete / checkpoint diversi.
#
# DISEGNO CONTROLLATO
#   Tutti gli input di simulazione (dataset, prior, rigid body, config,
#   equilibrated.npz) vengono da UNA directory canonica, BASE. Solo il
#   modello varia (MODEL_DIR/MODEL). Cosi' fra due run cambia esattamente
#   una cosa: i pesi.
#
# USO
#   MODE=priors bash 35_run_thermo_production.sh
#   MODE=ml     bash 35_run_thermo_production.sh                  # modello di BASE
#   MODE=ml MODEL_DIR=.../C1_snapshots_64x2 MODEL=C1_model.ep20.pt \
#               bash 35_run_thermo_production.sh                  # uno snapshot
#
# Variabili: STEPS (10000), DT (0.001), LOG_INTERVAL (20), RUN_DIR, BASE,
#            MODEL_DIR (alias: SRC), MODEL. 82 ms/step: 10000 step ~ 14 min.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT="$(cd "${TEL22_DIR}/../.." && pwd)"

MODE="${MODE:-ml}"
STEPS="${STEPS:-10000}"
DT="${DT:-0.001}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
BASE="${BASE:-${TEL22_DIR}/diagnostics/smoke/antiparallel_long_40ep}"
MODEL_DIR="${MODEL_DIR:-${SRC:-${BASE}}}"
MODEL="${MODEL:-tel22_model.pt}"
CONFIG_NAME="${CONFIG_NAME:-tel22_training_config_pipeline40.json}"
PYRESSO="${PYRESSO:-${ROOT}/espresso/build/pypresso}"

tag="${MODEL%.pt}"
RUN_DIR="${RUN_DIR:-${TEL22_DIR}/diagnostics/thermo/${MODE}_${tag}_$(( STEPS / 1000 ))kstep}"

# ── preflight ────────────────────────────────────────────────────────────────
[[ -x "${PYRESSO}" ]] || { echo "[ERROR] pypresso assente: ${PYRESSO}" >&2; exit 2; }
for f in tel22_dataset.bin cg_priors.json rigid_bodies_info.json equilibrated.npz "${CONFIG_NAME}"; do
    [[ -s "${BASE}/${f}" ]] || { echo "[ERROR] assente in BASE: ${BASE}/${f}" >&2; exit 2; }
done
[[ -s "${MODEL_DIR}/${MODEL}" ]] || { echo "[ERROR] modello assente: ${MODEL_DIR}/${MODEL}" >&2; exit 2; }
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "[ERROR] directory non vuota: ${RUN_DIR}" >&2
    echo "        mv '${RUN_DIR}' '${RUN_DIR}.STALE'   oppure   RUN_DIR=... " >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"

# ── input: tutto da BASE, solo il modello da MODEL_DIR ───────────────────────
for f in tel22_dataset.bin cg_priors.json rigid_bodies_info.json equilibrated.npz "${CONFIG_NAME}"; do
    cp "${BASE}/${f}" "${RUN_DIR}/"
done
cp "${MODEL_DIR}/${MODEL}" "${RUN_DIR}/"
EXTRA=()
if [[ -s "${MODEL_DIR}/${MODEL}.manifest.json" ]]; then
    cp "${MODEL_DIR}/${MODEL}.manifest.json" "${RUN_DIR}/"
else
    EXTRA+=(--allow_missing_model_manifest)
    echo "[INFO] ${MODEL} senza manifest: uso --allow_missing_model_manifest"
fi
# Il checkpoint equilibrated.npz registra lo SHA del modello che lo ha
# prodotto. Testare un altro checkpoint dalla STESSA configurazione iniziale e'
# deliberato (confronto controllato), quindi il mismatch di provenienza va
# permesso -- ma SOLO quello: dopo la run si verifica che nessun altro campo
# (architettura, box, particelle, dataset, prior) sia in disaccordo.
DELIBERATE_MODEL_SWAP=0
if [[ "$(cd "${MODEL_DIR}" && pwd)/${MODEL}" != "$(cd "${BASE}" && pwd)/tel22_model.pt" ]]; then
    DELIBERATE_MODEL_SWAP=1
    EXTRA+=(--allow_checkpoint_mismatch)
    echo "[INFO] modello diverso da quello di equilibrated.npz: mismatch di provenienza"
    echo "       del solo modello permesso; ogni altro mismatch fara' fallire la run."
fi
if [[ "${MODE}" == "priors" ]]; then
    EXTRA+=(--disable_ml)
    echo "[INFO] MODE=priors -> rete DISATTIVATA (controllo solo-prior)"
else
    echo "[INFO] MODE=ml -> prior + rete PaiNN | modello: ${MODEL_DIR}/${MODEL}"
fi

# ── run ──────────────────────────────────────────────────────────────────────
cd "${RUN_DIR}"
echo "[INFO] ${STEPS} step, dt=${DT}, campione ogni ${LOG_INTERVAL} step -> $(( STEPS / LOG_INTERVAL )) frame"
set +e   # con -e la pipeline fallita ucciderebbe lo script prima del controllo sotto
PYTHONUNBUFFERED=1 "${PYRESSO}" "${ROOT}/simulation/run_cg_md.py" \
    --model "${MODEL}" \
    --config "${CONFIG_NAME}" \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel22_dataset.bin \
    --checkpoint equilibrated.npz \
    --steps "${STEPS}" --dt "${DT}" \
    --device auto --neighbor_search link-cell \
    --energy_file energy.csv --no_vtf \
    --sample_npz samples.npz --log_interval "${LOG_INTERVAL}" \
    --out_checkpoint production_final.npz \
    ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee production_stdout.log
sim_status=${PIPESTATUS[0]}
set -e

# Se la simulazione e' morta senza produrre campioni, la directory contiene
# solo input copiati e un log d'errore: rinominarla in .FAILED evita che il
# guard "directory non vuota" blocchi il tentativo successivo.
if [[ ${sim_status} -ne 0 && ! -s samples.npz ]]; then
    cd ..
    failed="${RUN_DIR}.FAILED_$(date +%H%M%S)"
    mv "${RUN_DIR}" "${failed}"
    echo "[FAIL] simulazione uscita con stato ${sim_status} senza campioni." >&2
    echo "       Directory rinominata: ${failed}" >&2
    echo "       Puoi rilanciare lo stesso comando senza spostare nulla." >&2
    exit "${sim_status}"
fi

# ── verifica post-run: solo il modello poteva differire ─────────────────────
if [[ "${DELIBERATE_MODEL_SWAP}" == "1" ]]; then
    unexpected="$(grep -E '^\[WARNING\]   - ' production_stdout.log \
        | grep -vE '^\[WARNING\]   - (model_sha256|model_manifest_sha256):' || true)"
    if [[ -n "${unexpected}" ]]; then
        echo "[FATAL] mismatch di provenienza NON previsto (oltre al modello):" >&2
        echo "${unexpected}" >&2
        echo "        La run non e' un confronto controllato. Risultati da scartare." >&2
        exit 3
    fi
    echo "[CHECK] unico mismatch: il modello (deliberato). Confronto controllato valido."
fi
[[ -s samples.npz ]] || { echo "[FATAL] samples.npz assente: la simulazione non e' arrivata in fondo." >&2; exit 4; }
echo "[OK] ${MODE}/${MODEL}: campioni in ${RUN_DIR}/samples.npz"
