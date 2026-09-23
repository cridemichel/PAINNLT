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
VELOCITY_SEED="${VELOCITY_SEED:-314159}"
# MODEL: come in 05, per equilibrare un checkpoint diverso dal "best".
MODEL="${MODEL:-tel26_model.pt}"
# CHECKPOINT: il nome dello stato equilibrato.  Era fisso su equilibrated.npz,
# e con piu' produzioni in parallelo nella stessa directory -- lo sweep sugli
# otto checkpoint -- i job si sovrascrivevano a vicenda: la produzione di un
# modello poteva partire dallo stato equilibrato di un altro, e il risultato
# sarebbe stato mescolato in modo indistinguibile.  Uno per modello.
CHECKPOINT="${CHECKPOINT:-equilibrated.npz}"
# CLASSICAL=1 salta le due fasi ML dell'equilibrazione (con e senza force cap)
# e produce uno stato equilibrato con i soli prior.  E' il punto di partenza
# del controllo solo-prior: un equilibrato ML non serve a quel confronto, e se
# e' il modello a scaldare il sistema quello stato sarebbe gia' compromesso.
phase_args=()
[ -n "${CLASSICAL:-}" ] && phase_args+=(--steps_ml_capped 0 --steps_ml_uncapped 0)
# EQ_DT: il passo dell'equilibrazione.  equilibrate.py usa 0.002 ps di default
# mentre 05_run_espresso.sh produce a 0.001: lo stato viene cosi' generato con
# un passo DOPPIO rispetto a quello che poi deve accettarlo.  Se l'equilibrato
# arriva caldo -- E_kin molto sopra ~(3 N_mol + 3 N_rigidi)/2 * kT -- questa e'
# la prima cosa da stringere, perche' la fase 4 gira senza force cap e con un
# termostato debole (gamma = 1 ps^-1), che non dissipa quanto un integratore
# instabile inietta.
[ -n "${EQ_DT:-}" ] && phase_args+=(--dt "${EQ_DT}")
NEIGHBOR_SEARCH="${NEIGHBOR_SEARCH:-link-cell}"

cd "${SCRIPT_DIR}"

for path in "${MODEL}" tel26_training_config.json cg_priors.json rigid_bodies_info.json tel26_dataset.bin; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing required input: ${path}" >&2
        exit 1
    fi
done

"${PYRESSO}" "${FRAMEWORK_ROOT}/simulation/equilibrate.py" \
    --model "${MODEL}" \
    ${phase_args[@]+"${phase_args[@]}"} \
    --config tel26_training_config.json \
    --priors cg_priors.json \
    --rb_info rigid_bodies_info.json \
    --dataset tel26_dataset.bin \
    --out_checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    --neighbor_search "${NEIGHBOR_SEARCH}" \
    --kT 2.49 \
    --velocity_seed "${VELOCITY_SEED}"

echo "[DONE] ${CHECKPOINT}"
