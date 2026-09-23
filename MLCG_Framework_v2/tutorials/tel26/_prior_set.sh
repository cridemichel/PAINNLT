# shellcheck shell=bash
# Nomi dei file che dipendono dall'insieme di prior, per gli stadi 02-05.
#
# PERCHE'
#   Lo studio di fattibilita' confronta piu' insiemi di prior analitici sullo
#   stesso riferimento all-atom (Morse COM-COM del template, Morse sito-sito
#   sulle B3, FENE, ...).  Il dataset NON e' indipendente dai prior: le forze
#   che il modello impara sono il residuo F_AA - F_prior, quindi ogni insieme
#   ha il suo dataset, il suo cg_priors e i suoi modelli.  Con nomi fissi un
#   secondo insieme cancellerebbe il primo, e un modello allenato sui residui
#   di un insieme finirebbe simulato sopra i prior di un altro -- senza errori,
#   con una fisica sbagliata.
#
#   PRIOR_SET vuoto      tel26_topology.json          tel26_dataset.bin
#                        cg_priors.json               rigid_bodies_info.json
#   PRIOR_SET=b3morse    tel26_topology.b3morse.json  tel26_b3morse_dataset.bin
#                        cg_priors.b3morse.json       rigid_bodies_info.b3morse.json
#
#   I modelli prendono il nome dall'insieme e dalla variante di training:
#   tel26_b3morse_d64_model.pt.  04 e 05 ricavano l'insieme dal manifest del
#   modello (dataset_path), e si rifiutano di mettere un residuo ML sopra prior
#   diversi da quelli su cui e' stato allenato.

prior_set_files() {
    local s="${1:-}"
    TOPOLOGY_JSON="tel26_topology${s:+.${s}}.json"
    DATASET_BIN="tel26${s:+_${s}}_dataset.bin"
    PRIORS_JSON="cg_priors${s:+.${s}}.json"
    RB_INFO_JSON="rigid_bodies_info${s:+.${s}}.json"
}

# L'insieme di prior su cui e' stato allenato un modello, dal dataset che il
# suo manifest registra.  Stampa "?" se il manifest manca o non e' leggibile.
prior_set_of_model() {
    python3 - "${1}.manifest.json" <<'PYEOF' 2>/dev/null || echo "?"
import json, os, re, sys
name = os.path.basename(json.load(open(sys.argv[1]))["dataset_path"])
m = re.fullmatch(r"tel26(?:_(.+))?_dataset\.bin", name)
print("?" if m is None else (m.group(1) or ""))
PYEOF
}

# Da chiamare in 04/05 dopo aver fissato MODEL.  ML_ACTIVE=1 se il residuo ML
# entra davvero nella dinamica (non CLASSICAL / DISABLE_ML).
resolve_prior_set() {
    local ml_active="${1}" trained
    trained="$(prior_set_of_model "${MODEL}")"
    if [ -z "${PRIOR_SET+x}" ]; then
        if [ "${trained}" = "?" ]; then
            PRIOR_SET=""
        else
            PRIOR_SET="${trained}"
        fi
    fi
    if [ "${ml_active}" = 1 ] && [ "${trained}" != "?" ] && [ "${trained}" != "${PRIOR_SET}" ]; then
        echo "[ERROR] ${MODEL} e' allenato sui residui dell'insieme '${trained:-canonico}'," >&2
        echo "        non su '${PRIOR_SET:-canonico}': il residuo ML ha senso solo sopra i" >&2
        echo "        prior che sono stati sottratti per allenarlo." >&2
        exit 1
    fi
    prior_set_files "${PRIOR_SET}"
    echo "[INFO] prior: ${PRIOR_SET:-canonico} (${PRIORS_JSON}, ${DATASET_BIN})"
}
