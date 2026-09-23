#!/usr/bin/env bash
# Training del modello TEL26.
#
#   bash 03_train_model.sh              config tel26_training_config.json  -> tel26_model.pt
#   RUN=d64 bash 03_train_model.sh      config tel26_training_config.d64.json -> tel26_d64_model.pt
#
# PERCHE' RUN E NON SOVRASCRIVERE LA CONFIG
#   La produzione valida l'architettura del modello contro la config con cui
#   lo si simula.  Sovrascrivere tel26_training_config.json con una variante
#   renderebbe inutilizzabili tutti i modelli gia' allenati con quella vecchia.
#   Ogni variante ha quindi config e modelli con nomi suoi, e 04/05 ricavano la
#   config giusta dal manifest del modello, che la registra in config_path.
#
# IL LOG
#   Il trainer scrive sempre cg_training_log.csv, nome fisso: un secondo
#   training cancellerebbe il primo.  Alla fine lo si copia accanto al modello.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAINER="${TRAINER:-${FRAMEWORK_ROOT}/training/build/train_painn}"
cd "${SCRIPT_DIR}"

# PRIOR_SET sceglie il dataset (i residui di quale insieme di prior) ed entra
# nel nome del modello: RUN=d64 PRIOR_SET=b3morse -> tel26_b3morse_d64_model.pt.
# shellcheck source=_prior_set.sh
source "${SCRIPT_DIR}/_prior_set.sh"
PRIOR_SET="${PRIOR_SET:-}"
prior_set_files "${PRIOR_SET}"
RUN="${RUN:-}"
CONFIG="tel26_training_config${RUN:+.${RUN}}.json"
MODEL_OUT="tel26${PRIOR_SET:+_${PRIOR_SET}}${RUN:+_${RUN}}_model.pt"

for path in "${DATASET_BIN}" "${CONFIG}"; do
    [ -f "${path}" ] || { echo "[ERROR] manca: ${path}" >&2; exit 1; }
done
[ -x "${TRAINER}" ] || { echo "[ERROR] trainer non eseguibile: ${TRAINER}" >&2; exit 1; }

# Un log lasciato da un training precedente non va perso.
if [ -f cg_training_log.csv ] && [ ! -f "${MODEL_OUT%.pt}.training_log.csv" ]; then
    mv cg_training_log.csv "cg_training_log.prima_di_${MODEL_OUT%.pt}.csv"
fi

echo "[INFO] dataset ${DATASET_BIN}, config ${CONFIG} -> ${MODEL_OUT}"
"${TRAINER}" "${DATASET_BIN}" "${MODEL_OUT}" "${CONFIG}"
cp cg_training_log.csv "${MODEL_OUT%.pt}.training_log.csv"
echo "[DONE] ${MODEL_OUT} (log in ${MODEL_OUT%.pt}.training_log.csv)"
