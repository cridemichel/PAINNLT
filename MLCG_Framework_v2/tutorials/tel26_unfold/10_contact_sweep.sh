#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=240G
#SBATCH -p dcgp_usr_prod
#SBATCH -A IscrB_G4MES_0
#SBATCH -t 04:00:00
#SBATCH -J csweep
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-csweep-%j.out
# Scansione (lam_tet, lam_oth, T) con i SOLI prior su una copia di TEL26, su un nodo DCGP.
#
# Per ogni coppia di fattori di scala dei contatti Morse (09_scale_contacts.py: tet = contatti
# dentro una tetrade, oth = stacking e loop) e ogni temperatura, una corsa da CG_STEPS passi
# partendo dal TEL26 nativo (primo frame del dataset t400), un core per corsa.  Alla fine
# 08_tetrad_melt.py per ogni coppia (etichette = T) e una tabella riassuntiva con P(F), P(U)
# contro T e la temperatura a cui P(F) scende sotto 1/2.
#
# USO (Leonardo):  sbatch $U/10_contact_sweep.sh
# Variabili: LT ("0.08 0.12 0.16 0.24"), LO ("0.08 0.16 0.24"), TS ("300 330 360 400 450"),
#            CG_STEPS (25000000 = 100 ns), LOG_INTERVAL (1250 = 5 ps), OUTDIR ($A/sweep),
#            ANALYZE_ONLY=1 per rifare solo l'analisi.
set -uo pipefail
A=${A:-/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold}
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=$R/tutorials/tel26
source $R/hpc/env_leonardo.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

LT=${LT:-"0.08 0.12 0.16 0.24"}
LO=${LO:-"0.08 0.16 0.24"}
TS=${TS:-"300 330 360 400 450"}
CG_STEPS=${CG_STEPS:-25000000}
LOG_INTERVAL=${LOG_INTERVAL:-1250}
OUTDIR=${OUTDIR:-$A/sweep}
MODEL=$T26/tel26_lp2_re1_it30.pt          # solo provenienza: PaiNN e' disattivato
CONFIG=$T26/$(python3 -c 'import json,sys,os;print(os.path.basename(json.load(open(sys.argv[1]))["config_path"]))' "$MODEL.manifest.json") || exit 1
BASE=$A/cg/cg_priors.lp2_1c.json
RBINFO=$A/cg/rigid_bodies_info.lp2_1c.t400.json
DATASET=$A/cg/tel26_lp2_1c_t400_dataset.bin
PYPRESSO=$R/espresso/build/pypresso
mkdir -p "$OUTDIR" && cd "$OUTDIR" || exit 1

tag_of() { printf "lt%.2f_lo%.2f" "$1" "$2"; }
run_args() {   # priors T nome passi
    local kt; kt=$(python3 -c "print(f'{0.008314462618*$2:.6f}')")
    echo --model "$MODEL" --config "$CONFIG" --disable_ml --priors "$1" --rb_info "$RBINFO" --dataset "$DATASET" \
         --steps "$4" --dt 0.004 --gamma 2 --kT "$kt" --init_kT "$kt" --device cpu --neighbor_search verlet \
         --thermostat_seed $(( $2 * 7 + 11 )) --velocity_seed $(( $2 * 13 + 5 )) \
         --log_interval "$LOG_INTERVAL" --energy_interval 10 --no_vtf \
         --sample_npz "$3.samples.npz" --energy_file "$3.energy.csv" --out_checkpoint "$3.state.npz"
}

if [[ -z "${ANALYZE_ONLY:-}" ]]; then
    mapfile -t CPUS < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
    jobs_list=()
    for lt in $LT; do for lo in $LO; do
        tag=$(tag_of $lt $lo); mkdir -p $tag
        python3 $U/09_scale_contacts.py --in $BASE --out $tag/cg_priors.json --lam-tet $lt --lam-oth $lo > $tag/scale.txt || exit 1
        for T in $TS; do jobs_list+=("$tag $T"); done
    done; done
    echo "[csweep] ${#jobs_list[@]} corse, ${#CPUS[@]} core, $CG_STEPS passi ciascuna (log ogni $LOG_INTERVAL)"
    (( ${#jobs_list[@]} <= ${#CPUS[@]} )) || { echo "[ERROR] piu' corse che core" >&2; exit 1; }

    # prova breve sul primo elemento: se pypresso non parte su questo nodo, ci si ferma subito
    set -- ${jobs_list[0]}
    "$PYPRESSO" $R/simulation/run_cg_md.py $(run_args $1/cg_priors.json $2 $1/smoke 2000) > $1/smoke.log 2>&1 \
        || { echo "[ERROR] prova breve fallita: vedi $OUTDIR/$1/smoke.log" >&2; tail -20 $1/smoke.log; exit 1; }
    echo "[csweep] prova breve ok: $(grep -c 'Step' $1/smoke.log) righe di log"

    pids=(); i=0
    for item in "${jobs_list[@]}"; do
        set -- $item
        name=$1/T$2
        [[ -e $name.samples.npz ]] && { echo "[csweep] $name gia' fatto, salto"; continue; }
        taskset -c ${CPUS[$i]} "$PYPRESSO" $R/simulation/run_cg_md.py $(run_args $1/cg_priors.json $2 $name $CG_STEPS) \
            > $name.log 2>&1 &
        pids+=($!); i=$((i + 1))
    done
    fail=0
    for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
    echo "[csweep] corse finite, fallite: $fail"
fi

# ── analisi ──────────────────────────────────────────────────────────────────
for lt in $LT; do for lo in $LO; do
    tag=$(tag_of $lt $lo); specs=()
    for T in $TS; do [[ -s $tag/T$T.samples.npz ]] && specs+=("$T=$OUTDIR/$tag/T$T.samples.npz"); done
    (( ${#specs[@]} )) || continue
    echo; echo "=== $tag ==="; cat $tag/scale.txt
    python3 $U/08_tetrad_melt.py "${specs[@]}" --plot $tag/melt --json $tag/melt.json
done; done
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "lt*_lo*", "melt.json"))):
    tag = os.path.basename(os.path.dirname(f))
    runs = json.load(open(f))["runs"]
    T = sorted((float(k), v) for k, v in runs.items())
    tf = next((f"{a:.0f}-{b:.0f}" for (a, va), (b, vb) in zip(T, T[1:]) if va["P_F"] >= 0.5 > vb["P_F"]),
              "<" + f"{T[0][0]:.0f}" if T[0][1]["P_F"] < 0.5 else ">" + f"{T[-1][0]:.0f}")
    rows.append((tag, T, tf))
if rows:
    Ts = [t for t, _ in rows[0][1]]
    print("\n  P(F) / P(U) contro T (seconda meta' di ogni corsa);  T_F = intervallo dove P(F) passa 1/2")
    print(f"  {'insieme':16s}" + "".join(f"{t:>12.0f}" for t in Ts) + "      T_F (K)")
    for tag, T, tf in rows:
        print(f"  {tag:16s}" + "".join(f"   {v['P_F']:4.2f}/{v['P_U']:4.2f}" for _, v in T) + f"   {tf:>10s}")
    print("  Tm sperimentale (U al 50%): 328 K;  F al 50%: 307 K")
PY
