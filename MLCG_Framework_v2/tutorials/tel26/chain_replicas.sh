#!/usr/bin/env bash
# Produzione lunga: catena di job, ognuno con REPLICAS corse sulla stessa GPU.
#
# Ogni segmento e' un job run_replicas_per_gpu.sh che riparte dagli stati
# finali del segmento precedente (velocita' conservate), con dipendenza
# afterok: se un segmento fallisce, i successivi restano in coda e non partono
# (si cancellano con scancel).  Da lanciare sul login node.
#
# Nomi: <PREFIX>_s01, <PREFIX>_s02, ...; file <PREFIX>_sNN_r<i>.samples.npz,
#   .energy.csv, .state.npz, .log in tutorials/tel26.
#
# Esempio (Booster, 4 segmenti da 5 ns per replica = 2 x 20 ns):
#   PREFIX=prod_re1 START_FROM=rep1ns_re1_b NSEG=4 bash chain_replicas.sh
# Continuare una catena gia' fatta (dal segmento 5 in poi):
#   PREFIX=prod_re1 START_FROM=prod_re1_s04 FIRST_SEG=5 NSEG=4 bash chain_replicas.sh
#
# Variabili: PREFIX (obbligatoria), START_FROM (obbligatoria: prefisso degli
#   stati iniziali, <START_FROM>_r<i>.state.npz), NSEG (4), FIRST_SEG (1),
#   NS_PER_SEG (5), MODEL (tel26_lp2_re1_it30.pt), REPLICAS (2),
#   LOG_INTERVAL (250 = 1 ps a dt 4 fs), TIME (07:00:00), ACCOUNT
#   (IscrB_G4MES), PARTITION (boost_usr_prod), AFTER (job id da attendere
#   prima del primo segmento), DRY_RUN=1 per stampare i comandi senza sottometterli,
#   DISABLE_ML=1 per la stessa catena con i soli prior (modello solo per provenienza).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)/logs}"
: "${PREFIX:?serve PREFIX, il prefisso dei segmenti}"
: "${START_FROM:?serve START_FROM, il prefisso degli stati iniziali}"
NSEG="${NSEG:-4}"
FIRST_SEG="${FIRST_SEG:-1}"
NS_PER_SEG="${NS_PER_SEG:-5}"
MODEL="${MODEL:-tel26_lp2_re1_it30.pt}"
REPLICAS="${REPLICAS:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-250}"
TIME="${TIME:-07:00:00}"
ACCOUNT="${ACCOUNT:-IscrB_G4MES}"
PARTITION="${PARTITION:-boost_usr_prod}"
AFTER="${AFTER:-}"
DT_PS=0.004
CG_STEPS=$(python3 -c "print(round(${NS_PER_SEG} * 1000 / ${DT_PS}))")

for ((i = 0; i < REPLICAS; i++)); do
    if [[ ! -s "${SCRIPT_DIR}/${START_FROM}_r${i}.state.npz" && -z "${AFTER}" ]]; then
        echo "[ERROR] manca ${START_FROM}_r${i}.state.npz in ${SCRIPT_DIR}" >&2
        exit 1
    fi
done
mkdir -p "${LOG_DIR}"

echo "[chain] ${PREFIX}: segmenti ${FIRST_SEG}..$((FIRST_SEG + NSEG - 1)), ${NS_PER_SEG} ns x ${REPLICAS} replicas ciascuno"
echo "[chain] ${CG_STEPS} step per segmento, LOG_INTERVAL ${LOG_INTERVAL}, modello ${MODEL}, partenza da ${START_FROM}"

prev_name="${START_FROM}"
prev_job="${AFTER}"
for ((k = FIRST_SEG; k < FIRST_SEG + NSEG; k++)); do
    name="$(printf '%s_s%02d' "${PREFIX}" "${k}")"
    dep=()
    [[ -n "${prev_job}" ]] && dep=(--dependency="afterok:${prev_job}")
    wrap="cd ${SCRIPT_DIR} && NAME=${name} CONTINUE_FROM=${prev_name} MODEL=${MODEL} REPLICAS=${REPLICAS} CG_STEPS=${CG_STEPS} LOG_INTERVAL=${LOG_INTERVAL} ${DISABLE_ML:+DISABLE_ML=1} bash ${SCRIPT_DIR}/run_replicas_per_gpu.sh"
    cmd=(sbatch --parsable -A "${ACCOUNT}" -p "${PARTITION}" --gres=gpu:1 --cpus-per-task=8
         --time="${TIME}" -J "tel26_${name}" ${dep[@]+"${dep[@]}"}
         -o "${LOG_DIR}/slurm-tel26_${name}-%j.out" --wrap "${wrap}")
    if [[ -n "${DRY_RUN:-}" ]]; then
        printf '%q ' "${cmd[@]}"; echo
        job="DRY${k}"
    else
        job="$("${cmd[@]}")"
        job="${job%%;*}"
    fi
    echo "[chain] ${name}: job ${job}${prev_job:+ (dopo ${prev_job})}, da ${prev_name}"
    prev_name="${name}"
    prev_job="${job}"
done
echo "[chain] ultimo segmento: ${prev_name} (job ${prev_job})"
