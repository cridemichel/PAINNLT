#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -p boost_usr_prod
#SBATCH -A IscrB_G4MES
#SBATCH -t 03:00:00
#SBATCH -J benchbox
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-benchbox-%j.out
# Costo di una scatola con N TEL26 (S(q) totale): ms/passo, ns/giorno, memoria, quota PaiNN.
#
# Per ogni N di NS: 25_build_box.py (concentrazione CONC_mM, meta' F e meta' U dalle corse a una
# copia di pop6 md_it21 a 330 K), poi run_cg_md.py con i contatti come legami (bonded-switched),
# grafo PaiNN sulla GPU, STEPS passi da velocita' nuove.  Dal profilo PaiNN (--painn_profile_report,
# primi 20 passi esclusi): ms/passo complessivi e di PaiNN; il resto e' ESPResSo (CPU, un processo).
# Memoria: picco RSS (/usr/bin/time -v).  Esito: una riga [BENCH] per N.
# SKIN (nm) > 0 attiva la lista di Verlet del grafo PaiNN (MLCG_PAINN_VERLET_SKIN, plugin ricompilato):
# senza, le scatole oltre 20 000 siti (~230 copie) si saltano.
#
# USO (Leonardo):  sbatch $U/26_bench_box.sh            (NS="10 30 100 300", SKIN=0.3)
# Variabili: NS, CONC_mM (0.5), STEPS (2000), SKIN (0.3), MODEL, OUT ($A/bench_box)
set -uo pipefail
A=${A:-/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold}
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=$R/tutorials/tel26
source $R/hpc/env_leonardo.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MLCG_PAINN_GRAPH=device
NS=${NS:-"10 30 100 300"}
CONC_mM=${CONC_mM:-0.5}
STEPS=${STEPS:-2000}
SKIN=${SKIN:-0.3}
MODEL=${MODEL:-$A/relent_pop/pop6/pop6_it21.pt}
OUT=${OUT:-$A/bench_box}
SRC=$A/relent_pop/pop6/md_it21
PYPRESSO=$R/espresso/build/pypresso
CONFIG=$T26/$(python3 -c 'import json,sys,os;print(os.path.basename(json.load(open(sys.argv[1]))["config_path"]))' "$MODEL.manifest.json") || exit 1
PRI1=$A/thermo_val/f/cg_priors.json
RB=$A/cg/rigid_bodies_info.lp2_1c.t400.json
TEMPLATE=$A/cg/tel26_lp2_1c_t400_dataset.bin
T=330
kt=$(python3 -c "print(f'{0.008314462618*$T:.6f}')")
# conformazioni: F dal lignaggio F a 318 K (a 330 K il modello e' gia' tutto U), U dal lignaggio U a 338 K
SRC_F=${SRC_F:-$SRC/F_T318.samples.npz}
SRC_U=${SRC_U:-$SRC/U_T338.samples.npz}
for f in $MODEL $CONFIG $PRI1 $RB $TEMPLATE $SRC_F $SRC_U $PYPRESSO; do
    [[ -s $f ]] || { echo "[ERROR] manca $f" >&2; exit 1; }
done
export MLCG_PAINN_VERLET_SKIN=$SKIN
mkdir -p $OUT && cd $OUT || exit 1
echo "[bench] N: $NS; $CONC_mM mM; $STEPS passi; skin $SKIN nm; $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
for N in $NS; do
    D=$OUT/n$N
    mkdir -p $D
    python3 $U/25_build_box.py --n-copies $N --conc-mM $CONC_mM --pu 0.5 \
        --samples $SRC_F $SRC_U --skip-ps 100 \
        --priors $PRI1 --template $TEMPLATE --out-priors $D/cg_priors.json --out-dataset $D/dataset.bin \
        --seed $N > $D/build.log 2>&1 || { tail -3 $D/build.log; continue; }
    sites=$((N * 86))
    if [[ $SKIN == 0 ]] && (( sites > 20000 )); then
        echo "[bench] N $N: $sites siti, oltre il limite del grafo senza lista di Verlet: salto"; continue
    fi
    TIMEV=(); [[ -x /usr/bin/time ]] && TIMEV=(/usr/bin/time -v)
    t0=$(date +%s.%N)
    ${TIMEV[@]+"${TIMEV[@]}"} $PYPRESSO $R/simulation/run_cg_md.py --model $MODEL --config $CONFIG --priors $D/cg_priors.json \
        --rb_info $RB --dataset $D/dataset.bin --pair_specific_morse_runtime bonded-switched \
        --steps $STEPS --dt 0.004 --gamma 2 --kT $kt --init_kT $kt --velocity_seed $((N + 7)) --thermostat_seed $((N + 11)) \
        --device cuda --neighbor_search verlet --log_interval 500 --energy_interval 10 --no_vtf \
        --energy_file $D/energy.csv --painn_profile_report $D/profile.json > $D/run.log 2> $D/time.log
    rc=$?
    t1=$(date +%s.%N)
    if (( rc != 0 )); then echo "[bench] N $N: run_cg_md fallito (rc $rc)"; tail -5 $D/run.log; tail -3 $D/time.log; continue; fi
    python3 - "$N" "$D" "$t0" "$t1" "$STEPS" <<'PYEOF'
import json, re, sys
N, D, t0, t1, steps = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
p = json.load(open(f"{D}/profile.json"))
ms = p["integration"]["wall_ms_per_step"]; painn = p["integration"]["painn_total_ms_per_force_call"]
calls = p["integration"]["painn_force_calls_per_requested_step"]
rss = next((int(m.group(1)) for m in re.finditer(r"Maximum resident set size \(kbytes\): (\d+)", open(f"{D}/time.log").read())), 0)
box = json.load(open(f"{D}/dataset.json"))
ns_day = 0.004 * 86400 / ms if ms > 0 else 0.0                        # dt 4 fs: ns/giorno = 345.6 / ms
print(f"[BENCH] N {N:4d}  siti {N * 86:6d}  L {box['box_nm']:.1f} nm  {ms:8.2f} ms/passo "
      f"(PaiNN {painn * calls:7.2f}, resto {ms - painn * calls:7.2f})  {ns_day:6.2f} ns/giorno  "
      f"RSS {rss / 1024**2:.1f} GB  totale {t1 - t0:.0f} s (avvio compreso)")
PYEOF
done
