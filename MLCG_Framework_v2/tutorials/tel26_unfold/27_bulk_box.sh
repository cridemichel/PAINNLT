#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -p boost_usr_prod
#SBATCH -A IscrB_G4MES
#SBATCH -t 24:00:00
#SBATCH -J bulk
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-bulk-%j.out
# Sistema bulk: N TEL26 CG-ML (pop6_it21) alla concentrazione CONC_mM e alla temperatura T, per la S(q).
#
# 1. 25_build_box.py: prior a N copie e configurazione iniziale con la miscela F/U a P_U(T) del fit a
#    due stati della validazione (VALID/validation.json); conformazioni dalle corse di validazione alla T
#    piu' vicina (rami chiuso e aperto), con riserva F a 318 K e U a 342 K se a quella T manca uno stato.
# 2. run_cg_md.py a blocchi di CHUNK_NS: contatti nativi come legami (bonded-switched), grafo PaiNN sulla
#    GPU con lista di Verlet (MLCG_PAINN_VERLET_SKIN), campioni ogni LOG_PS; ogni blocco riparte dallo
#    stato del precedente.  Si ferma a NS_TOTAL o prima della fine del job (MAX_HOURS): basta risottomettere
#    con le stesse variabili, riprende dall'ultimo blocco completo.
# Analisi: 28_sq_box.py sulle cartelle di uscita.
#
# USO (Leonardo):  sbatch --export=ALL,T=322,SEED=1 $U/27_bulk_box.sh
# Variabili: T (obbligatoria), N (300), CONC_mM (0.5), SEED (1), NS_TOTAL (20), CHUNK_NS (1), LOG_PS (10),
#            SKIN (0.3), MODEL ($A/relent_pop/pop6/pop6_it21.pt), VALID ($A/melt_painn/pop6_it21),
#            MAX_HOURS (23.3), OUTDIR ($A/bulk/N<N>_c<CONC>_T<T>_s<SEED>)
set -uo pipefail
A=${A:-/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold}
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=$R/tutorials/tel26
source $R/hpc/env_leonardo.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MLCG_PAINN_GRAPH=device
: "${T:?serve T (K), es. --export=ALL,T=322}"
N=${N:-300}
CONC_mM=${CONC_mM:-0.5}
SEED=${SEED:-1}
NS_TOTAL=${NS_TOTAL:-20}
CHUNK_NS=${CHUNK_NS:-1}            # 300 copie: ~2-3 ns/giorno su una A100 (26_bench_box.sh), un blocco < 12 h
LOG_PS=${LOG_PS:-10}
SKIN=${SKIN:-0.3}
MODEL=${MODEL:-$A/relent_pop/pop6/pop6_it21.pt}
VALID=${VALID:-$A/melt_painn/pop6_it21}
MAX_HOURS=${MAX_HOURS:-23.3}
OUTDIR=${OUTDIR:-$A/bulk/N${N}_c${CONC_mM}_T${T}_s${SEED}}
PYPRESSO=$R/espresso/build/pypresso
CONFIG=$T26/$(python3 -c 'import json,sys,os;print(os.path.basename(json.load(open(sys.argv[1]))["config_path"]))' "$MODEL.manifest.json") || exit 1
PRI1=$A/thermo_val/f/cg_priors.json
RB=$A/cg/rigid_bodies_info.lp2_1c.t400.json
TEMPLATE=$A/cg/tel26_lp2_1c_t400_dataset.bin
for f in $MODEL $CONFIG $PRI1 $RB $TEMPLATE $VALID/validation.json $PYPRESSO; do
    [[ -s $f ]] || { echo "[ERROR] manca $f" >&2; exit 1; }
done
export MLCG_PAINN_VERLET_SKIN=$SKIN
mkdir -p $OUTDIR && cd $OUTDIR || exit 1
exec 9> .lock
flock -n 9 || { echo "[ERROR] $OUTDIR e' gia' in uso da un altro job" >&2; exit 1; }
T_START=$(date +%s)
kt=$(python3 -c "print(f'{0.008314462618*$T:.6f}')")
DT=0.004
CHUNK_STEPS=$(python3 -c "print(int(round($CHUNK_NS * 1000 / $DT)))")
LOG_STEPS=$(python3 -c "print(int(round($LOG_PS / $DT)))")
NCHUNK=$(python3 -c "import math; print(math.ceil($NS_TOTAL / $CHUNK_NS))")
echo "[bulk] $OUTDIR: N $N, $CONC_mM mM, T $T K, seme $SEED, $NS_TOTAL ns in blocchi da $CHUNK_NS ns, campioni ogni $LOG_PS ps, skin $SKIN nm"

if [[ ! -s dataset.bin ]]; then
    Tn=$(ls $VALID/chiuso/T*.samples.npz | sed 's#.*/T\([0-9.]*\)\.samples\.npz#\1#' \
         | python3 -c "import sys; ts=[float(x) for x in sys.stdin.read().split()]; print(f'{min(ts, key=lambda x: abs(x - $T)):g}')")
    near=($VALID/chiuso/T$Tn.samples.npz $VALID/aperto/T$Tn.samples.npz)
    build() {
        python3 $U/25_build_box.py --n-copies $N --conc-mM $CONC_mM --pu-json $VALID/validation.json --T $T \
            --samples "$@" --skip-ps 5000 --priors $PRI1 --template $TEMPLATE \
            --out-priors cg_priors.json --out-dataset dataset.bin --seed $SEED
    }
    echo "[bulk] conformazioni dalle corse di validazione a $Tn K"
    build "${near[@]}" > build.log 2>&1 || {
        echo "[bulk] a $Tn K manca uno stato: aggiungo la riserva F (318 K) e U (342 K)"
        build "${near[@]}" $VALID/chiuso/T318.samples.npz $VALID/aperto/T342.samples.npz > build.log 2>&1 \
            || { cat build.log; exit 1; }
    }
    grep "^\[box\]" build.log
fi

common=(--model $MODEL --config $CONFIG --priors cg_priors.json --rb_info $RB --dataset dataset.bin
        --pair_specific_morse_runtime bonded-switched --dt $DT --gamma 2 --kT $kt --device cuda
        --neighbor_search verlet --log_interval $LOG_STEPS --energy_interval 10 --no_vtf)
last_dur=0
for k in $(seq 1 $NCHUNK); do
    name=$(printf "chunk%03d" $k)
    [[ -s $name.state.npz && -s $name.samples.npz ]] && continue
    now=$(date +%s)
    if (( last_dur > 0 )) && python3 -c "import sys; sys.exit(0 if $now + 1.15 * $last_dur > $T_START + $MAX_HOURS * 3600 else 1)"; then
        echo "[bulk] il blocco $k non ci sta nel job (ultimo: $((last_dur / 60)) min): risottometti"
        break
    fi
    if (( k == 1 )); then
        init=(--init_kT $kt --velocity_seed $((SEED * 1000 + 7)))
    else
        init=(--checkpoint $(printf "chunk%03d" $((k - 1))).state.npz)
    fi
    rm -f $name.tmp.samples.npz $name.tmp.state.npz
    $PYPRESSO $R/simulation/run_cg_md.py "${common[@]}" "${init[@]}" --steps $CHUNK_STEPS \
        --thermostat_seed $((SEED * 1000 + k)) --energy_file $name.energy.csv \
        --sample_npz $name.tmp.samples.npz --out_checkpoint $name.tmp.state.npz > $name.log 2>&1 \
        || { echo "[ERROR] blocco $k fallito:"; tail -8 $name.log; exit 1; }
    mv $name.tmp.samples.npz $name.samples.npz && mv $name.tmp.state.npz $name.state.npz
    last_dur=$(( $(date +%s) - now ))
    echo "[bulk] $(date '+%F %T') blocco $k/$NCHUNK ($CHUNK_NS ns) in $((last_dur / 60)) min"
done
done_n=$(ls chunk*.state.npz 2>/dev/null | wc -l)
echo "[bulk] blocchi completi: $done_n/$NCHUNK"
