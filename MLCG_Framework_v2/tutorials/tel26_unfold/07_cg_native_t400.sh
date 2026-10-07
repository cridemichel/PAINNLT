#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrB_G4MES
#SBATCH -t 08:00:00
#SBATCH -J cg_t400
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-cg_t400-%j.out
# Modello CG nativo (lp2 + re1_it30, allenato a 300 K su 10 copie) su UNA copia a T_K (400 K).
#
# Primo confronto col riferimento AA a 400 K (dataset di 06): quanto il modello nativo
# sovrastabilizza il quadruplex quando si alza solo la temperatura del termostato.  I prior
# sono quelli a 300 K tagliati a una copia (05_slice_copy.py); il residuo PaiNN e' identico.
# Serve anche come campione CG a 400 K per --check-gradient a due temperature.
#
# REPLICAS corse in parallelo sulla stessa GPU (core divisi con taskset), ognuna con semi
# propri e velocita' iniziali Maxwell-Boltzmann a T_K.  Stato iniziale: primo frame del
# dataset t400 (TEL26 nativo, scatola 9.77 nm).
#
# USO (Leonardo):  sbatch $U/07_cg_native_t400.sh
#   controllo con i soli prior:   sbatch --export=ALL,DISABLE_ML=1,NAME=pri_t400 $U/07_cg_native_t400.sh
#   continuazione:                sbatch --export=ALL,CONTINUE_FROM=nat_t400,NAME=nat_t400_b $U/07_cg_native_t400.sh
# Variabili: NAME (nat_t400), T_K (400), REPLICAS (2), CG_STEPS (2500000 = 10 ns a 4 fs),
#            LOG_INTERVAL (125 passi = 0.5 ps; per corse lunghe 1250 = 5 ps),
#            MODEL, OUTDIR ($A/cgmd_t400), DISABLE_ML, CONTINUE_FROM, SEED_BASE (1000)
set -uo pipefail
A=${A:-/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold}
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
T26=$R/tutorials/tel26
source $R/hpc/env_leonardo.sh

NAME=${NAME:-nat_t400}
T_K=${T_K:-400}
REPLICAS=${REPLICAS:-2}
CG_STEPS=${CG_STEPS:-2500000}
SEED_BASE=${SEED_BASE:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-125}
export PYTHONUNBUFFERED=1   # log leggibile durante la corsa
MODEL=${MODEL:-$T26/tel26_lp2_re1_it30.pt}
OUTDIR=${OUTDIR:-$A/cgmd_t400}
CONTINUE_FROM=${CONTINUE_FROM:-}
PRIORS=$A/cg/cg_priors.lp2_1c.json
RBINFO=$A/cg/rigid_bodies_info.lp2_1c.t400.json
DATASET=$A/cg/tel26_lp2_1c_t400_dataset.bin
PYPRESSO=$R/espresso/build/pypresso
KT=$(python3 -c "print(f'{0.008314462618*$T_K:.6f}')")

# config di training dal manifest del modello (percorso relativo alla cartella tel26)
CONFIG=$T26/$(python3 -c 'import json,sys,os;print(os.path.basename(json.load(open(sys.argv[1]))["config_path"]))' "$MODEL.manifest.json") || exit 1
for f in "$MODEL" "$CONFIG" "$PRIORS" "$RBINFO" "$DATASET" "$PYPRESSO"; do
    [[ -e "$f" ]] || { echo "[ERROR] manca $f" >&2; exit 1; }
done
mkdir -p "$OUTDIR" && cd "$OUTDIR" || exit 1
echo "[cg_t400] $NAME: T=${T_K} K (kT ${KT}), ${REPLICAS} repliche x ${CG_STEPS} passi (log ogni ${LOG_INTERVAL}), modello $(basename $MODEL), config $(basename $CONFIG), DISABLE_ML=${DISABLE_ML:-0}"

# core effettivamente assegnati dal job (non 0..N-1: dipende dal nodo)
mapfile -t CPUS < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
per=$(( ${#CPUS[@]} / REPLICAS )); (( per > 0 )) || per=1
pids=()
for ((i = 0; i < REPLICAS; i++)); do
    for s in samples.npz energy.csv state.npz; do
        [[ -e ${NAME}_r$i.$s ]] && { echo "[ERROR] ${NAME}_r$i.$s esiste: scegli un altro NAME" >&2; exit 1; }
    done
    args=(--model "$MODEL" --config "$CONFIG" --priors "$PRIORS" --rb_info "$RBINFO" --dataset "$DATASET"
          --steps "$CG_STEPS" --dt 0.004 --gamma 2 --kT "$KT" --device cuda --neighbor_search verlet
          --thermostat_seed $((SEED_BASE + i)) --log_interval "$LOG_INTERVAL" --energy_interval 10 --no_vtf
          --sample_npz ${NAME}_r$i.samples.npz --energy_file ${NAME}_r$i.energy.csv
          --out_checkpoint ${NAME}_r$i.state.npz)
    [[ -n "${DISABLE_ML:-}" ]] && args+=(--disable_ml)
    if [[ -n "$CONTINUE_FROM" ]]; then
        [[ -s ${CONTINUE_FROM}_r$i.state.npz ]] || { echo "[ERROR] manca ${CONTINUE_FROM}_r$i.state.npz" >&2; exit 1; }
        args+=(--checkpoint ${CONTINUE_FROM}_r$i.state.npz)
    else
        args+=(--init_kT "$KT" --velocity_seed $((SEED_BASE + 100 + i)))
    fi
    cores="$(IFS=,; echo "${CPUS[*]:$((i * per)):${per}}")"
    echo "[cg_t400] r$i: core $cores"
    MLCG_PAINN_GRAPH=device OMP_NUM_THREADS=$per taskset -c "$cores" \
        "$PYPRESSO" $R/simulation/run_cg_md.py "${args[@]}" > ${NAME}_r$i.log 2>&1 &
    pids+=($!)
done
rc=0
for ((i = 0; i < REPLICAS; i++)); do
    wait ${pids[$i]} || { echo "[cg_t400] replica $i FALLITA: vedi $OUTDIR/${NAME}_r$i.log"; rc=1; }
done
ls -la "$OUTDIR"/${NAME}_r*
exit $rc
