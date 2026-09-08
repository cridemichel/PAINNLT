#!/usr/bin/env bash
# Attende la fine del training C3 (vincolo spettrale 4.0) e lancia subito
# la valutazione dei suoi checkpoint.
#
# PERCHE' SERVE UN RUN_DIR ESPLICITO
#   Lo script 39 nomina gli snapshot C1_model.epN.pt qualunque sia il config,
#   quindi quelli di C3 hanno gli stessi nomi di quelli di C1 e differiscono
#   solo per la directory. Lo script 35 deriva RUN_DIR dal NOME del modello:
#   senza RUN_DIR esplicito, C3-ep20 finirebbe in ml_C1_model.ep20_10kstep,
#   che gia' esiste da C1 -> il guard "directory non vuota" lo bloccherebbe,
#   o peggio i due risultati diventerebbero indistinguibili.
#
# ORDINE DEI TEST
#   ep40 per primo: e' il test di stabilita' (C1-ep40 e' esploso a 3.6 ps).
#   ep20 secondo: e' l'epoca dove C1 da' il miglior fold (22.5% di legami H),
#   quindi e' il confronto strutturale a parita' di epoca.
#
# USO
#   bash 41_c3_eval_after_training.sh [PID_DEL_TRAINING]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMOKE="${TEL22_DIR}/diagnostics/smoke/C3_spectral4_64x2"
LOG="${TEL22_DIR}/diagnostics/thermo/41_c3_eval.log"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

say() { echo "[41] $(date '+%H:%M:%S') $*"; }

# ── 1. attesa della fine del training ────────────────────────────────────────
TRAIN_PID="${1:-}"
if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    say "training C3 in corso (pid ${TRAIN_PID}); attendo."
    while kill -0 "${TRAIN_PID}" 2>/dev/null; do sleep 30; done
    say "training terminato."
else
    say "nessun training attivo con pid '${TRAIN_PID}': procedo."
fi

# Il trainer scrive l'ultimo snapshot poco prima di uscire; lascio sedimentare.
sleep 10

# ── 2. verifica degli snapshot ───────────────────────────────────────────────
missing=0
for EP in 40 20; do
    if [[ ! -s "${SMOKE}/C1_model.ep${EP}.pt" ]]; then
        say "ERRORE: snapshot assente: ${SMOKE}/C1_model.ep${EP}.pt"
        missing=1
    fi
done
if (( missing )); then
    say "snapshot presenti: $(ls -1 "${SMOKE}"/C1_model.ep*.pt 2>/dev/null | wc -l | tr -d ' ')"
    ls -1 "${SMOKE}"/C1_model.ep*.pt 2>/dev/null | sed 's/^/      /'
    say "interrompo: senza ep40 il test di stabilita' non ha senso."
    exit 2
fi
say "snapshot ep40 e ep20 presenti."

# ── 3. le due produzioni, in sequenza ────────────────────────────────────────
for EP in 40 20; do
    RUN_DIR="${TEL22_DIR}/diagnostics/thermo/ml_C3_ep${EP}_10kstep"
    if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
        say "SALTO ep${EP}: ${RUN_DIR} gia' popolata."
        continue
    fi
    say "avvio produzione C3 ep${EP} -> $(basename "${RUN_DIR}") (~14 min)"
    MODE=ml \
    MODEL_DIR="${SMOKE}" \
    MODEL="C1_model.ep${EP}.pt" \
    RUN_DIR="${RUN_DIR}" \
    STEPS=10000 \
        bash "${SCRIPT_DIR}/35_run_thermo_production.sh"
    status=$?
    if (( status == 0 )); then
        say "ep${EP}: COMPLETATA."
    else
        # Un'uscita non-zero qui e' un RISULTATO, non un guasto: il guardrail
        # di sicurezza (max_f>10000, E_kin>5000, min_dist<0.15) ferma la
        # simulazione quando il modello esplode. E' cio' che ep40 deve dirci.
        say "ep${EP}: uscita con stato ${status} (probabile guardrail = esplosione)."
        d="${RUN_DIR}"
        [[ -d "${d}" ]] || d="$(ls -d "${RUN_DIR}".FAILED_* 2>/dev/null | tail -1)"
        [[ -n "${d}" && -s "${d}/production_stdout.log" ]] && \
            grep -aE "CRITICAL|Step [0-9]+/" "${d}/production_stdout.log" | tail -3 | sed 's/^/      /'
    fi
done

say "FINITO. Risultati in diagnostics/thermo/ml_C3_ep{40,20}_10kstep/"
