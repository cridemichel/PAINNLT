#!/bin/bash
# Ciclo di allenamento a ENTROPIA RELATIVA del residuo PaiNN (train_relent).
#
# Ogni iterazione n:
#   1. simula il modello it(n-1) sopra i prior (run_cg_md.py), ripartendo dallo
#      stato finale della corsa precedente, e scarta i primi SKIP_PS ps;
#   2. converte i campioni nel formato del dataset (samples_to_cg_bin.py);
#   3. train_relent: passi di gradiente beta(<dU/dtheta>_AA - <dU/dtheta>_CG),
#      con ripesatura dei campioni finche' ESS/N >= RE_ESS_MIN, e salva lo
#      stato con il Delta S_rel di holdout piu' basso -> modello it(n).
#
# La catena si RIPRENDE da sola: rilanciata con gli stessi TAG e PRIOR_SET,
# salta le iterazioni gia' fatte (modello it(n) presente) e riparte dalla
# prima mancante.  Il file di ogni passo si scrive per ultimo e per rinomina,
# quindi un job interrotto non lascia un'iterazione a meta' scambiata per
# completa.
#
# Si ferma dopo NITER iterazioni, o prima se per STOP_AFTER_NO_GAIN iterazioni
# consecutive nessun passo abbassa S_rel sull'holdout (convergenza, o campioni
# troppo rumorosi per vedere il guadagno: allungare CG_STEPS).
#
# Uso (tipico, dai soli prior lp1 con correzione ML nulla):
#   PRIOR_SET=lp1 START_CHECKPOINT=equilibrated_priors_lp1.npz TAG=re0 \
#       bash 06_relent.sh
#
# Variabili principali:
#   PRIOR_SET         insieme di prior (obbligatorio: la rete e' un residuo sopra quelli)
#   START_CHECKPOINT  stato equilibrato di partenza (obbligatorio)
#   CONFIG            architettura della rete (default: config d64)
#   START_MODEL       modello di partenza; vuoto = inizializzazione a zero (U_ML == 0)
#   TAG               nome della catena (default relent)
#   NITER             iterazioni (default 10)
#   CG_STEPS, CG_DT   lunghezza della simulazione per iterazione (default 25000 x 0.001 ps)
#   LOG_INTERVAL      passi fra due campioni (default 100 = 0.1 ps)
#   SKIP_PS           ps scartati dopo il cambio di modello (default 5)
#   GAMMA             attrito di Langevin (default 20: tau = m/gamma ~ 15 ps, come le corse g20)
#   RE_STEPS, RE_LR, RE_WD, RE_BATCH, RE_ESS_MIN, RE_HOLDOUT, RE_EVAL_EVERY
#                     parametri di train_relent
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -x "${FRAMEWORK_ROOT}/espresso/build/pypresso" ]]; then
    DEFAULT_PYPRESSO="${FRAMEWORK_ROOT}/espresso/build/pypresso"
else
    DEFAULT_PYPRESSO="pypresso"
fi
PYRESSO="${PYRESSO:-${DEFAULT_PYPRESSO}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RELENT_BIN="${RELENT_BIN:-${FRAMEWORK_ROOT}/training/build/train_relent}"
DEVICE="${DEVICE:-auto}"

: "${PRIOR_SET:?indica PRIOR_SET, insieme di prior sopra cui allenare il residuo (es. lp1)}"
: "${START_CHECKPOINT:?indica START_CHECKPOINT, lo stato equilibrato da cui partire}"
CONFIG="${CONFIG:-tel26_training_config.d64.json}"
START_MODEL="${START_MODEL:-}"
TAG="${TAG:-relent}"
NITER="${NITER:-10}"
CG_STEPS="${CG_STEPS:-25000}"
CG_DT="${CG_DT:-0.001}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SKIP_PS="${SKIP_PS:-5}"
GAMMA="${GAMMA:-20}"
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH:-link-cell}"
FINAL_SAMPLES="${FINAL_SAMPLES:-1}"
STOP_AFTER_NO_GAIN="${STOP_AFTER_NO_GAIN:-2}"
RE_STEPS="${RE_STEPS:-200}"
RE_LR="${RE_LR:-1e-3}"
RE_WD="${RE_WD:-0}"
RE_BATCH="${RE_BATCH:-8}"
RE_ESS_MIN="${RE_ESS_MIN:-0.5}"
RE_HOLDOUT="${RE_HOLDOUT:-0.2}"
RE_EVAL_EVERY="${RE_EVAL_EVERY:-10}"

cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/_prior_set.sh"
prior_set_files "${PRIOR_SET}"
for path in "${CONFIG}" "${PRIORS_JSON}" "${RB_INFO_JSON}" "${DATASET_BIN}" "${START_CHECKPOINT}"; do
    [ -f "${path}" ] || { echo "[ERROR] manca: ${path}" >&2; exit 1; }
done
[ -x "${RELENT_BIN}" ] || { echo "[ERROR] train_relent non compilato: ${RELENT_BIN}" >&2; exit 1; }
[ -z "${START_MODEL}" ] || [ -f "${START_MODEL}" ] || { echo "[ERROR] manca: ${START_MODEL}" >&2; exit 1; }

# Campionare solo dopo SKIP_PS: run_cg_md vuole un multiplo di LOG_INTERVAL.
SKIP_STEPS=$("${PYTHON_BIN}" -c "
import sys
dt, skip, li, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
s = int(round(skip / dt)) // li * li
if s >= n: sys.exit('SKIP_PS non lascia campioni: allunga CG_STEPS')
print(s)" "${CG_DT}" "${SKIP_PS}" "${LOG_INTERVAL}" "${CG_STEPS}")

base="tel26_${PRIOR_SET}_${TAG}"
model_of()   { printf "%s_it%02d.pt" "${base}" "$1"; }
samples_of() { printf "samples_%s_it%02d.npz" "${base}" "$1"; }
state_of()   { printf "state_%s_it%02d.npz" "${base}" "$1"; }
cgbin_of()   { printf "relent_cg_%s_it%02d.bin" "${base}" "$1"; }

manifest() {
    "${PYTHON_BIN}" "${FRAMEWORK_ROOT}/training/create_model_manifest.py" \
        --model "$1" --config "${CONFIG}" --dataset "${DATASET_BIN}" >/dev/null
}

# Una catena per cartella e per nome alla volta.
exec 9> ".relent_${base}.lock"
flock -n 9 || { echo "[ERROR] la catena ${base} e' gia' in corso" >&2; exit 1; }

# --- iterazione 0: il modello di partenza ---
m0="$(model_of 0)"
if [ ! -f "${m0}" ]; then
    if [ -n "${START_MODEL}" ]; then
        cp "${START_MODEL}" "${m0}.tmp" && mv "${m0}.tmp" "${m0}"
        echo "[INFO] partenza da ${START_MODEL}"
    else
        "${RELENT_BIN}" --config "${CONFIG}" --aa "${DATASET_BIN}" --zero-init \
            --steps 0 --out "${m0}" --report "${m0%.pt}.relent.json"
    fi
fi
[ -f "${m0}.manifest.json" ] || manifest "${m0}"

# Simula il modello it(k): campioni e stato finale, per rinomina a fine corsa.
simulate() {
    local k="$1" ckpt_in="$2"
    local model samples state
    model="$(model_of "$k")"; samples="$(samples_of "$k")"; state="$(state_of "$k")"
    if [ -f "${samples}" ] && [ -f "${state}" ]; then
        echo "[INFO] campioni di ${model} gia' presenti: ${samples}"
        return 0
    fi
    local extra=()
    [ -n "${GAMMA}" ] && extra+=(--gamma "${GAMMA}")
    echo "[INFO] $(date '+%F %T') MD di ${model} da ${ckpt_in}: ${CG_STEPS} passi, campioni dopo ${SKIP_STEPS}"
    # --allow_checkpoint_mismatch: lo stato viene dalla corsa del modello
    # precedente, quindi l'hash del modello non coincide per costruzione.
    "${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/run_cg_md.py" \
        --model "${model}" --config "${CONFIG}" \
        --priors "${PRIORS_JSON}" --rb_info "${RB_INFO_JSON}" --dataset "${DATASET_BIN}" \
        --checkpoint "${ckpt_in}" --allow_checkpoint_mismatch \
        --steps "${CG_STEPS}" --dt "${CG_DT}" --kT 2.49 \
        --device "${DEVICE}" --neighbor_search "${NEIGHBOR_SEARCH}" \
        --log_interval "${LOG_INTERVAL}" --no_vtf \
        --energy_file "energy_${base}_it$(printf %02d "$k").csv" \
        --sample_npz "${samples}.tmp.npz" --sample_start_step "${SKIP_STEPS}" \
        --out_checkpoint "${state}.tmp.npz" \
        ${extra[@]+"${extra[@]}"}
    mv "${state}.tmp.npz" "${state}"
    mv "${samples}.tmp.npz" "${samples}"
}

no_gain=0
last=0
for n in $(seq 1 "${NITER}"); do
    prev=$((n - 1))
    next_model="$(model_of "$n")"
    if [ -f "${next_model}" ] && [ -f "${next_model}.manifest.json" ]; then
        last="$n"
        continue
    fi
    if [ "${prev}" -eq 0 ]; then ckpt="${START_CHECKPOINT}"; else ckpt="$(state_of $((prev - 1)))"; fi
    simulate "${prev}" "${ckpt}"

    cgbin="$(cgbin_of "${prev}")"
    [ -f "${cgbin}" ] || "${PYTHON_BIN}" "${FRAMEWORK_ROOT}/preprocessing/samples_to_cg_bin.py" \
        --template "${DATASET_BIN}" --samples "$(samples_of "${prev}")" --out "${cgbin}"

    report="${next_model%.pt}.relent.json"
    echo "[INFO] $(date '+%F %T') train_relent: $(model_of "${prev}") -> ${next_model}"
    "${RELENT_BIN}" --config "${CONFIG}" --aa "${DATASET_BIN}" --cg "${cgbin}" \
        --in "$(model_of "${prev}")" --out "${next_model}.tmp.pt" --report "${report}" \
        --steps "${RE_STEPS}" --lr "${RE_LR}" --weight-decay "${RE_WD}" \
        --batch-aa "${RE_BATCH}" --batch-cg "${RE_BATCH}" --ess-min "${RE_ESS_MIN}" \
        --holdout-frac "${RE_HOLDOUT}" --eval-every "${RE_EVAL_EVERY}" \
        --seed "$((42 + n))" --device "${DEVICE}"
    mv "${next_model}.tmp.pt" "${next_model}"
    manifest "${next_model}"
    last="$n"

    summary=$("${PYTHON_BIN}" -c "
import json, sys
r = json.load(open(sys.argv[1]))
print(r['best_step'], r['best_dS'], r['stop_reason'], r['ess_last'], round(r['seconds']))" "${report}")
    read -r best_step best_ds stop ess secs <<<"${summary}"
    echo "[RELENT-ITER] it$(printf %02d "$n") passo_migliore=${best_step} dS_holdout=${best_ds}" \
         "arresto=${stop} ESS/N=${ess} (${secs} s)"
    if [ "${best_step}" -eq 0 ]; then
        no_gain=$((no_gain + 1))
        if [ "${no_gain}" -ge "${STOP_AFTER_NO_GAIN}" ]; then
            echo "[CONVERGENZA] ${no_gain} iterazioni senza guadagno su S_rel: catena ferma a it$(printf %02d "$n")"
            break
        fi
    else
        no_gain=0
    fi
done

# Campioni dell'ultimo modello, per l'analisi strutturale.
if [ "${FINAL_SAMPLES}" = 1 ] && [ "${last}" -ge 1 ]; then
    simulate "${last}" "$(state_of $((last - 1)))"
fi

echo "[DONE] catena ${base}: ultimo modello $(model_of "${last}")"
echo "[INFO] g(r) per iterazione:"
runs=""
for k in $(seq 0 "${last}"); do
    [ -f "$(samples_of "$k")" ] && runs+=" it$(printf %02d "$k")=$(samples_of "$k")"
done
echo "  bash hpc/submit_leonardo.sh analysis SYSTEM=tel26 PRIOR_SET=${PRIOR_SET} RUNS=\"${runs# }\""
