#!/usr/bin/env bash
# Selezione del checkpoint su osservabile strutturale, con MD brevi.
#
# PERCHE' NON C'E' UN'ALTERNATIVA A SIMULARE
#   Su questo problema nessuna metrica di validazione ordina i modelli.  La loss
#   sulle forze istantanee e' ~98% rumore irriducibile; la skill contro il
#   predittore-zero ha un tetto di ~1.7% e non discrimina; e l'accordo sulla
#   curva di forza media - per entrambe le strade provate, dalle forze e dalla
#   g(r) di riferimento - ordina AL CONTRARIO.  Validato su quattro modelli:
#
#     modello   g(r) B3-B3 misurata   accordo da forze   da g(r)
#     D=32               0.587              0.000          0.035
#     D=64               0.746  <- best     0.357          0.358
#     D=128              0.286  <- worst    0.480  <- max  0.568  <- max
#
#   La ragione e' che tutte quelle quantita' si misurano sulle configurazioni di
#   RIFERIMENTO, mentre TEL22 fallisce per deriva dell'ensemble del modello.
#   Vederla richiede di campionare il modello.
#
# PERCHE' COSTA POCO COMUNQUE
#   Non serve una produzione lunga.  La sovrapposizione g(r) sul canale B3-B3
#   riproduce a 1 ps l'ordinamento che si misura a 10 ps, con separazioni ampie
#   (D=64 0.880 contro D=128 0.325).  1000 passi sono ~90 s per checkpoint.
#
# USO
#   MODEL_DIR=<dir con gli snapshot> bash 45_select_checkpoint.sh [ep3 ep6 ...]
#   Senza argomenti prende tutti i *.ep*.pt della directory.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEL22_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${TEL22_DIR}}"
BASE="${BASE:-${TEL22_DIR}/diagnostics/smoke/antiparallel_long_40ep}"
STEPS="${STEPS:-1000}"
OUT_ROOT="${OUT_ROOT:-${TEL22_DIR}/diagnostics/select}"
PY="${PYTHON_BIN:-python3}"

# mapfile non esiste in bash 3.2, che e' quello che macOS spedisce come
# /usr/bin/env bash: si accumula in un array con un ciclo portabile.
models=()
if [[ $# -gt 0 ]]; then
    for tag in "$@"; do
        for f in "${MODEL_DIR}"/*."${tag}".pt "${MODEL_DIR}/${tag}"; do
            [[ -s "$f" ]] && models+=("$f")
        done
    done
else
    for f in "${MODEL_DIR}"/*.ep*.pt; do
        [[ -s "$f" ]] && models+=("$f")
    done
fi
if [[ ${#models[@]} -eq 0 ]]; then
    echo "[ERROR] nessun checkpoint trovato in ${MODEL_DIR}" >&2
    exit 2
fi
echo "[INFO] ${#models[@]} checkpoint, ${STEPS} passi ciascuno (~$(( STEPS / 11 )) s)"

mkdir -p "${OUT_ROOT}"
declare -a produced=()
for m in "${models[@]}"; do
    name="$(basename "${m}" .pt)"
    rd="${OUT_ROOT}/${name}_${STEPS}step"
    if [[ -s "${rd}/samples.npz" ]]; then
        echo "[INFO] ${name}: gia' presente, riuso"
        produced+=("${name}=${rd}/samples.npz")
        continue
    fi
    rm -rf "${rd}"
    echo "[INFO] ${name}: produzione..."
    if MODE=ml MODEL_DIR="$(dirname "${m}")" MODEL="$(basename "${m}")" \
       RUN_DIR="${rd}" STEPS="${STEPS}" BASE="${BASE}" \
       bash "${SCRIPT_DIR}/35_run_thermo_production.sh" > "${rd%.*}.log" 2>&1
    then
        produced+=("${name}=${rd}/samples.npz")
    else
        # Un'uscita non-zero e' un RISULTATO: il guardrail ferma la simulazione
        # quando il modello esplode, e un modello che esplode e' escluso.
        echo "[INFO] ${name}: uscita non-zero (guardrail = instabile). Escluso."
    fi
done

if [[ ${#produced[@]} -eq 0 ]]; then
    echo "[ERROR] nessuna produzione utilizzabile" >&2
    exit 2
fi

echo
echo "[INFO] Confronto strutturale sul canale B3-B3 (legami di Hoogsteen)."
echo "       E' il canale che porta l'informazione sul quadruplex: la g(r)"
echo "       totale mescola i tipi e nasconde l'errore."
"${PY}" "${SCRIPT_DIR}/44_tel22_rdf.py" "${BASE}/tel22_dataset.bin" "${produced[@]}"
