#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=240G
#SBATCH -p dcgp_usr_prod
#SBATCH -A IscrB_G4MES_0
#SBATCH -t 06:00:00
#SBATCH -J thval
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-thval-%j.out
# Validazione diretta dei prior dipendenti da T fittati da 11_fit_thermal.py (soli prior, una copia).
#
# Per ogni insieme (h_tet, T0_tet, h_oth, T0_oth) 09_scale_contacts.py scrive UN file di prior
# termici (D_H, D_S; D = null): run_cg_md.py risolve D(T) alla T del termostato.  Ogni insieme
# gira a tutte le T di TS, partendo dal nativo (ramo "chiuso") e da uno stato aperto (ramo
# "aperto", START_OPEN), REPS repliche indipendenti per punto, un core per corsa.
# Default: 2 insiemi x 14 T x 2 rami x 2 repliche = 112 corse = un nodo DCGP.
# Alla fine: 08_tetrad_melt.py per insieme e ramo (repliche unite con '+'), poi
# 14_thermal_check.py: popolazioni contro esperimento e contro l'interpolazione della griglia,
# isteresi, Tm, dH di van 't Hoff (= <U_H>_U - <U_H>_F anche con Hamiltoniana dipendente da T).
#
# USO (Leonardo):  sbatch $U/13_validate_thermal.sh
# Variabili: SETS ("nome:mappa:h_tet:T0_tet:h_oth:T0_oth ..."), TS (295..360 ogni 5 K), REPS (2),
#            BRANCHES ("chiuso aperto"), START_OPEN ($A/sweep2/lt0.08_lo0.08/T450.state.npz),
#            CG_STEPS (25000000 = 100 ns), LOG_INTERVAL (1250), OUTDIR ($A/thermo_val),
#            FITDIR ($A/sweep2: fit_thermal.json / fit_thermal_f.json per il confronto), ANALYZE_ONLY=1.
set -uo pipefail
A=${A:-/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold}
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=$R/tutorials/tel26
source $R/hpc/env_leonardo.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SETS=${SETS:-"f1:f1:0.5120:426.67:0.15910:481.64 f:f:0.09309:2560.2:1.09415:373.66"}
TS=${TS:-"295 300 305 310 315 320 325 330 335 340 345 350 355 360"}
REPS=${REPS:-2}
BRANCHES=${BRANCHES:-"chiuso aperto"}
START_OPEN=${START_OPEN:-$A/sweep2/lt0.08_lo0.08/T450.state.npz}
CG_STEPS=${CG_STEPS:-25000000}
LOG_INTERVAL=${LOG_INTERVAL:-1250}
OUTDIR=${OUTDIR:-$A/thermo_val}
FITDIR=${FITDIR:-$A/sweep2}
MODEL=$T26/tel26_lp2_re1_it30.pt          # solo provenienza: PaiNN e' disattivato
CONFIG=$T26/$(python3 -c 'import json,sys,os;print(os.path.basename(json.load(open(sys.argv[1]))["config_path"]))' "$MODEL.manifest.json") || exit 1
BASE=$A/cg/cg_priors.lp2_1c.json
RBINFO=$A/cg/rigid_bodies_info.lp2_1c.t400.json
DATASET=$A/cg/tel26_lp2_1c_t400_dataset.bin
PYPRESSO=$R/espresso/build/pypresso
[[ " $BRANCHES " != *" aperto "* || -s "$START_OPEN" ]] || { echo "[ERROR] START_OPEN=$START_OPEN non esiste" >&2; exit 1; }
mkdir -p "$OUTDIR" && cd "$OUTDIR" || exit 1

run_args() {   # priors T nome passi seme start
    local kt; kt=$(python3 -c "print(f'{0.008314462618*$2:.6f}')")
    echo --model "$MODEL" --config "$CONFIG" --disable_ml --priors "$1" --rb_info "$RBINFO" --dataset "$DATASET" \
         --steps "$4" --dt 0.004 --gamma 2 --kT "$kt" --init_kT "$kt" --device cpu --neighbor_search verlet \
         --thermostat_seed "$5" --velocity_seed $(( $5 + 7919 )) \
         --log_interval "$LOG_INTERVAL" --energy_interval 10 --no_vtf \
         --sample_npz "$3.samples.npz" --energy_file "$3.energy.csv" --out_checkpoint "$3.state.npz" \
         ${6:+--checkpoint "$6" --allow_checkpoint_mismatch}
}

if [[ -z "${ANALYZE_ONLY:-}" ]]; then
    mapfile -t CPUS < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
    jobs_list=(); si=0
    for s in $SETS; do
        IFS=: read -r name map ht t0t ho t0o <<< "$s"
        mkdir -p $name
        python3 $U/09_scale_contacts.py --in $BASE --out $name/cg_priors.json \
            --h-tet $ht --t0-tet $t0t --h-oth $ho --t0-oth $t0o --report-T 295,310,325,340,355 > $name/scale.txt || exit 1
        echo "== $name (mappa $map)"; cat $name/scale.txt
        bi=0
        for b in $BRANCHES; do
            mkdir -p $name/$b
            for T in $TS; do for r in $(seq 0 $((REPS - 1))); do
                jobs_list+=("$name $b $T $r $(( T * 7 + 11 + 1000 * r + 100 * bi + 10000 * si ))")
            done; done
            bi=$((bi + 1))
        done
        si=$((si + 1))
    done
    echo "[thval] ${#jobs_list[@]} corse, ${#CPUS[@]} core, $CG_STEPS passi ciascuna"
    (( ${#jobs_list[@]} <= ${#CPUS[@]} )) || { echo "[ERROR] piu' corse che core" >&2; exit 1; }

    # prova breve (ramo aperto se c'e': controlla anche checkpoint + prior termici)
    set -- ${jobs_list[-1]}
    st=""; [[ $2 == aperto ]] && st=$START_OPEN
    mkdir -p $OUTDIR/$1/wd_smoke
    ( cd $OUTDIR/$1/wd_smoke && "$PYPRESSO" $R/simulation/run_cg_md.py \
        $(run_args $OUTDIR/$1/cg_priors.json $3 $OUTDIR/$1/smoke 2000 $5 $st) ) > $1/smoke.log 2>&1 \
        || { echo "[ERROR] prova breve fallita: vedi $OUTDIR/$1/smoke.log" >&2; tail -20 $1/smoke.log; exit 1; }
    grep -m1 "prior termici" $1/smoke.log || { echo "[ERROR] prior termici non risolti nella prova breve" >&2; exit 1; }

    pids=(); i=0
    for item in "${jobs_list[@]}"; do
        set -- $item
        name=$OUTDIR/$1/$2/T$3_r$4
        [[ -e $name.samples.npz ]] && { echo "[thval] $name gia' fatto, salto"; continue; }
        st=""; [[ $2 == aperto ]] && st=$START_OPEN
        mkdir -p $OUTDIR/$1/$2/wd_T$3_r$4
        ( cd $OUTDIR/$1/$2/wd_T$3_r$4 && exec taskset -c ${CPUS[$i]} "$PYPRESSO" $R/simulation/run_cg_md.py \
            $(run_args $OUTDIR/$1/cg_priors.json $3 $name $CG_STEPS $5 $st) ) > $name.log 2>&1 &
        pids+=($!); i=$((i + 1))
    done
    fail=0
    for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
    echo "[thval] corse finite, fallite: $fail"
fi

# ── analisi ──────────────────────────────────────────────────────────────────
check_args=()
for s in $SETS; do
    IFS=: read -r name map ht t0t ho t0o <<< "$s"
    for b in $BRANCHES; do
        specs=()
        for T in $TS; do
            f=$(ls $OUTDIR/$name/$b/T${T}_r*.samples.npz 2>/dev/null | paste -sd+)
            [[ -n "$f" ]] && specs+=("$T=$f")
        done
        (( ${#specs[@]} )) || continue
        echo; echo "=== $name / $b ==="
        python3 $U/08_tetrad_melt.py "${specs[@]}" --plot $name/$b/melt --json $name/$b/melt.json
    done
    fj=$FITDIR/fit_thermal.json; [[ $map == f ]] && fj=$FITDIR/fit_thermal_f.json
    check_args+=(--set "$name=$OUTDIR/$name:$map${fj:+:$fj}")
done
python3 $U/14_thermal_check.py "${check_args[@]}" --plot $OUTDIR/thermal_check --json $OUTDIR/thermal_check.json
