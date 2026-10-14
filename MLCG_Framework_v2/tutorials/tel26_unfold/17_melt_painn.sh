#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH -p boost_usr_prod
#SBATCH -A IscrB_G4MES
#SBATCH -t 24:00:00
#SBATCH -J melt_ml
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-melt_ml-%j.out
# Curva di fusione con PaiNN ATTIVO sopra i prior termici (una copia di TEL26).
#
# Come 13_validate_thermal.sh, ma col residuo PaiNN allenato a 300 K sopra lp2th (catena th0)
# e su GPU: per ogni T di TS e ogni ramo (chiuso = TEL26 nativo, aperto = START_OPEN) una corsa
# da CG_STEPS passi; le corse si distribuiscono a turno sulle 4 GPU del nodo, un core ciascuna.
# Prior: PRIORS (default $A/thermo_val/f/cg_priors.json = insieme f a una copia, lo stesso di
# lp2th per copia), risolti alla T del termostato.  Analisi: 08 con q mediato su 1 ns
# (melt_s1.json per ramo) e 14 contro l'esperimento (thermal_check_s1.*).
#
# USO (Leonardo):  sbatch $U/17_melt_painn.sh
#   continuazione (stessi stati finali, nuovo NAME):  CONTINUE=1 (riparte da T<T>.state.npz e scrive T<T>_b.*)
# Variabili: MODEL ($T26/tel26_lp2th_th0_it60.pt), TS ("300 315 325 330 335 340 350 365 380"),
#            BRANCHES ("chiuso aperto"), CG_STEPS (12500000 = 50 ns), LOG_INTERVAL (1250 = 5 ps),
#            OUTDIR ($A/melt_painn/<modello>), PRIORS, START_OPEN, ANALYZE_ONLY=1
set -uo pipefail
A=${A:-/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold}
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=$R/tutorials/tel26
source $R/hpc/env_leonardo.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MLCG_PAINN_GRAPH=device

MODEL=${MODEL:-$T26/tel26_lp2th_th0_it60.pt}
TAG=$(basename "$MODEL" .pt)
TS=${TS:-"300 315 325 330 335 340 350 365 380"}
BRANCHES=${BRANCHES:-"chiuso aperto"}
CG_STEPS=${CG_STEPS:-12500000}
LOG_INTERVAL=${LOG_INTERVAL:-1250}
OUTDIR=${OUTDIR:-$A/melt_painn/$TAG}
PRIORS=${PRIORS:-$A/thermo_val/f/cg_priors.json}
START_OPEN=${START_OPEN:-$A/sweep2/lt0.08_lo0.08/T450.state.npz}
CONTINUE=${CONTINUE:-}
SUF=""; [[ -n "$CONTINUE" ]] && SUF="_b"
RBINFO=$A/cg/rigid_bodies_info.lp2_1c.t400.json
DATASET=$A/cg/tel26_lp2_1c_t400_dataset.bin
PYPRESSO=$R/espresso/build/pypresso
CONFIG=$T26/$(python3 -c 'import json,sys,os;print(os.path.basename(json.load(open(sys.argv[1]))["config_path"]))' "$MODEL.manifest.json") || exit 1
for f in "$MODEL" "$CONFIG" "$PRIORS" "$RBINFO" "$DATASET" "$PYPRESSO"; do [[ -e $f ]] || { echo "[ERROR] manca $f" >&2; exit 1; }; done
[[ " $BRANCHES " != *" aperto "* || -s "$START_OPEN" ]] || { echo "[ERROR] START_OPEN=$START_OPEN non esiste" >&2; exit 1; }
python3 -c "import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if any('D_H' in b for b in p['bonds']) else 1)" "$PRIORS" \
    || { echo "[ERROR] $PRIORS non e' un insieme di prior termici" >&2; exit 1; }
mkdir -p "$OUTDIR" && cd "$OUTDIR" || exit 1
echo "[melt_ml] $TAG, prior $PRIORS, T: $TS, rami: $BRANCHES, $CG_STEPS passi${CONTINUE:+ (continuazione)}"

if [[ -z "${ANALYZE_ONLY:-}" ]]; then
    mapfile -t CPUS < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); (( NGPU > 0 )) || NGPU=1
    jobs_list=(); bi=0
    for b in $BRANCHES; do mkdir -p $b; for T in $TS; do jobs_list+=("$b $T $bi"); done; bi=$((bi + 1)); done
    (( ${#jobs_list[@]} <= ${#CPUS[@]} )) || { echo "[ERROR] piu' corse che core" >&2; exit 1; }
    echo "[melt_ml] ${#jobs_list[@]} corse su $NGPU GPU e ${#CPUS[@]} core"
    pids=(); i=0
    for item in "${jobs_list[@]}"; do
        set -- $item
        b=$1; T=$2; name=$OUTDIR/$b/T$T$SUF
        [[ -e $name.samples.npz ]] && { echo "[melt_ml] $name gia' fatto, salto"; continue; }
        kt=$(python3 -c "print(f'{0.008314462618*$T:.6f}')")
        seed=$(( T * 7 + 11 + 100 * $3 + ${CONTINUE:+5000}0 ))
        args=(--model "$MODEL" --config "$CONFIG" --priors "$PRIORS" --rb_info "$RBINFO" --dataset "$DATASET"
              --steps "$CG_STEPS" --dt 0.004 --gamma 2 --kT "$kt" --device cuda --neighbor_search verlet
              --thermostat_seed $seed --log_interval "$LOG_INTERVAL" --energy_interval 10 --no_vtf
              --sample_npz $name.samples.npz --energy_file $name.energy.csv --out_checkpoint $name.state.npz)
        if [[ -n "$CONTINUE" ]]; then
            [[ -s $OUTDIR/$b/T$T.state.npz ]] || { echo "[melt_ml] manca $OUTDIR/$b/T$T.state.npz, salto"; continue; }
            args+=(--checkpoint $OUTDIR/$b/T$T.state.npz)
        else
            args+=(--init_kT "$kt" --velocity_seed $(( seed + 7919 )))
            [[ $b == aperto ]] && args+=(--checkpoint "$START_OPEN" --allow_checkpoint_mismatch)
        fi
        mkdir -p $OUTDIR/$b/wd_T$T$SUF
        ( cd $OUTDIR/$b/wd_T$T$SUF && CUDA_VISIBLE_DEVICES=$(( i % NGPU )) exec taskset -c ${CPUS[$i]} \
            "$PYPRESSO" $R/simulation/run_cg_md.py "${args[@]}" ) > $name.log 2>&1 &
        pids+=($!); i=$((i + 1))
    done
    sleep 300
    echo "[melt_ml] dopo 5 min:"; for f in $OUTDIR/*/T*$SUF.log; do echo "  $(basename $(dirname $f))/$(basename $f): $(grep -c 'Step' $f) righe, $(grep -m1 -o 'prior termici a T = [0-9.]* K' $f)"; done
    fail=0
    for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
    echo "[melt_ml] corse finite, fallite: $fail"
fi

# ── analisi: 08 con q mediato su 1 ns per ramo, poi 14 contro l'esperimento ──
for b in $BRANCHES; do
    specs=()
    for T in $TS; do
        f=$(ls $OUTDIR/$b/T${T}.samples.npz $OUTDIR/$b/T${T}_b.samples.npz 2>/dev/null | paste -sd+)
        [[ -n "$f" ]] && specs+=("$T=$f")
    done
    (( ${#specs[@]} )) || continue
    echo; echo "=== $TAG / $b"
    python3 $U/08_tetrad_melt.py "${specs[@]}" --smooth-ns 1 --plot $b/melt_s1 --json $b/melt_s1.json
done
python3 $U/14_thermal_check.py --set "$TAG=$OUTDIR:f" --melt-name melt_s1.json \
    --plot $OUTDIR/thermal_check_s1 --json $OUTDIR/thermal_check_s1.json
