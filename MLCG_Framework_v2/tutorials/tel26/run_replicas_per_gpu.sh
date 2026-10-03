#!/usr/bin/env bash
# Produzione con piu' corse indipendenti sulla stessa GPU.
#
# Benchmark del 3/10 (bench_concurrency.sh, job 59228890): una corsa TEL26 non
# satura la A100 e la parte ESPResSo non usa i thread.  Due corse in parallelo
# sulla stessa GPU danno 1,5x i ns/giorno totali (quattro solo 1,6x), senza
# costo aggiuntivo sull'allocazione e senza toccare la fisica: ogni corsa e'
# una normale 05_run_espresso.sh, con i suoi semi e i suoi file.
#
# Ogni replica ha:
#   - un seme del termostato diverso (SEED_BASE + i);
#   - alla prima partenza, velocita' rigenerate da Maxwell-Boltzmann con un seme
#     diverso (--init_kT), cosi' le repliche non partono identiche;
#   - file propri: <NAME>_r<i>.samples.npz, .energy.csv, .state.npz, .log;
#   - core propri (taskset), divisi in parti uguali fra quelli del job.
#
# Prima partenza (tutte dallo stesso stato):
#   NAME=prod_re1 CHECKPOINT=equilibrated_priors_lp2_100ps.npz CG_STEPS=250000 \
#       bash run_replicas_per_gpu.sh
# Continuazione (ogni replica dal proprio stato, velocita' conservate):
#   NAME=prod_re1_b CONTINUE_FROM=prod_re1 CG_STEPS=250000 bash run_replicas_per_gpu.sh
#
# Su Leonardo (1 GPU + 8 core, cioe' un quarto di nodo):
#   sbatch -A IscrB_G4MES -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 \
#       --time=03:00:00 -J tel26_prod --wrap "NAME=... bash <percorso assoluto>/run_replicas_per_gpu.sh"
#
# Variabili proprie: NAME (obbligatoria), REPLICAS (2), SEED_BASE (1000),
#   CONTINUE_FROM (prefisso di una corsa precedente), KT_INIT (2.49),
#   VTF=1 per scrivere anche la traiettoria VTF.
# Tutte le altre (MODEL, CHECKPOINT, CG_STEPS, CG_DT, GAMMA, LOG_INTERVAL,
#   ENERGY_INTERVAL, DEVICE, MD_EXTRA_ARGS, ...) passano a 05_run_espresso.sh.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh" ]]; then
    # shellcheck disable=SC1091
    source "${FRAMEWORK_ROOT}/hpc/env_leonardo.sh"
fi
cd "${SCRIPT_DIR}"

: "${NAME:?serve NAME, il prefisso dei file di questa produzione}"
REPLICAS="${REPLICAS:-2}"
SEED_BASE="${SEED_BASE:-1000}"
KT_INIT="${KT_INIT:-2.49}"
CONTINUE_FROM="${CONTINUE_FROM:-}"
export DEVICE="${DEVICE:-cuda}"
RUN05="${SCRIPT_DIR}/05_run_espresso.sh"

for ((i = 0; i < REPLICAS; i++)); do
    for suffix in samples.npz energy.csv state.npz; do
        if [[ -e "${NAME}_r${i}.${suffix}" ]]; then
            echo "[ERROR] ${NAME}_r${i}.${suffix} esiste gia': scegli un altro NAME" >&2
            exit 1
        fi
    done
    if [[ -n "${CONTINUE_FROM}" && ! -s "${CONTINUE_FROM}_r${i}.state.npz" ]]; then
        echo "[ERROR] manca ${CONTINUE_FROM}_r${i}.state.npz per continuare la replica ${i}" >&2
        exit 1
    fi
done

mapfile -t CPUS < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
PER=$(( ${#CPUS[@]} / REPLICAS ))
if (( PER < 1 )); then
    echo "[ERROR] ${#CPUS[@]} core per ${REPLICAS} repliche" >&2
    exit 1
fi
echo "[replicas] ${NAME}: ${REPLICAS} repliche, ${PER} core ciascuna, device ${DEVICE}"

pids=()
for ((i = 0; i < REPLICAS; i++)); do
    cores="$(IFS=,; echo "${CPUS[*]:$((i * PER)):${PER}}")"
    seed=$((SEED_BASE + i))
    extra="--thermostat_seed ${seed} --energy_file ${SCRIPT_DIR}/${NAME}_r${i}.energy.csv --out_checkpoint ${SCRIPT_DIR}/${NAME}_r${i}.state.npz"
    if [[ -n "${CONTINUE_FROM}" ]]; then
        checkpoint="${CONTINUE_FROM}_r${i}.state.npz"
    else
        checkpoint="${CHECKPOINT:-equilibrated.npz}"
        # Stessa configurazione di partenza, velocita' diverse per replica.
        extra+=" --init_kT ${KT_INIT} --velocity_seed $((SEED_BASE + 100 + i)) --allow_checkpoint_mismatch"
    fi
    if [[ -n "${VTF:-}" ]]; then
        extra+=" --trajectory_file ${SCRIPT_DIR}/${NAME}_r${i}.vtf"
    else
        extra+=" --no_vtf"
    fi
    echo "[replicas] r${i}: core ${cores}, seme termostato ${seed}, stato iniziale ${checkpoint}"
    OMP_NUM_THREADS="${PER}" taskset -c "${cores}" env \
        CHECKPOINT="${checkpoint}" \
        SAMPLE_NPZ="${SCRIPT_DIR}/${NAME}_r${i}.samples.npz" \
        MD_EXTRA_ARGS="${extra} ${MD_EXTRA_ARGS:-}" \
        bash "${RUN05}" > "${SCRIPT_DIR}/${NAME}_r${i}.log" 2>&1 &
    pids+=("$!")
done

status=0
for ((i = 0; i < REPLICAS; i++)); do
    if wait "${pids[$i]}"; then
        echo "[replicas] r${i}: completata"
    else
        echo "[replicas] r${i}: FALLITA, vedi ${NAME}_r${i}.log" >&2
        status=1
    fi
done
echo "[replicas] campioni: $(ls ${NAME}_r*.samples.npz 2>/dev/null | tr '\n' ' ')"
echo "[replicas] struttura (script 46): repliche concatenate con '+', per esempio"
echo "    python3 46_tel26_structure.py tel26_lp1_dataset.bin prod=$(for ((i = 0; i < REPLICAS; i++)); do printf '%s' "${NAME}_r${i}.samples.npz#100:1000000"; ((i < REPLICAS - 1)) && printf '+'; done)"
echo "[replicas] tempi (script 47): una etichetta per replica; in 47 il '+' unisce tratti"
echo "    consecutivi della STESSA corsa, non repliche indipendenti."
exit "${status}"
