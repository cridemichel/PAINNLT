#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH -p boost_usr_prod
#SBATCH -A IscrB_G4MES
#SBATCH -t 24:00:00
#SBATCH -J relpop
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-relpop-%j.out
# Entropia relativa a 300 K + vincolo sulle popolazioni sperimentali (train_relent --pop).
#
# PERCHE' (19/10): il residuo th0_it60, allenato a 300 K sopra i prior termici lp2th, stabilizza
# F di 60-230 kJ/mol in energia libera (gap di energia F -> U 300-450): il TEL26 non fonde fino a
# 380 K (17), e nessun kappa globale (18) da' insieme Tm e larghezza.  Qui il residuo, testa
# readout_dT compresa, si allena insieme sulla struttura a 300 K e su ln(P_U/P_N) sperimentale a
# POP_TS, con le etichette di stato di 19_pop_labels.py (U = nessuna tetrade con q > 0.5, q mediato
# su 1 ns).
#
# OGNI ITERAZIONE n (modello it(n-1) -> it(n)), tutto su un nodo Booster:
#   1. MD in parallelo col modello it(n-1), ripartendo dagli stati finali dell'iterazione precedente:
#      - 300 K, insieme lp2th a 10 copie, MD_PS ps (campioni dopo SKIP_PS), come 06_relent.sh;
#      - per ogni T di POP_TS due corse a una copia (insieme f), lignaggio F (dal nativo) e U (da uno
#        stato aperto), POP_NS ns; alla prima iterazione da START_F / START_U con velocita' nuove;
#   2. dataset CG a 300 K (samples_to_cg_bin.py) e, per T, dataset + etichette (19_pop_labels.py);
#   3. train_relent --pop ... (lambda POP_LAMBDA, Huber POP_HUBER): modello it(n) col punteggio
#      Delta S_holdout + sum_T [P_T - P_T(theta_0)] piu' basso.
# Alla prima iterazione (CHECK=1) prima dell'allenamento si verifica il gradiente (--check-gradient).
# La catena riprende da sola (salta le iterazioni con modello e manifest) e si ferma prima della fine
# del job se l'iterazione successiva non ci sta (MAX_HOURS): basta risottomettere.
#
# USO (Leonardo, dopo il pull e la ricompilazione di train_relent):
#     sbatch $U/19_relent_pop.sh
#     sbatch --dependency=afterany:<jobid> $U/19_relent_pop.sh      # continuazione
# POP_TS si puo' cambiare a catena avviata: le T nuove partono da START_F / START_U.
# POP_F_TS (23/10): T a cui si vincola anche P_F (tre tetradi intatte; bersaglio nella mappa
# POP_F_MAP, default f1 = F + I1).  Le MD girano sull'unione di POP_TS e POP_F_TS.
# Variabili: TAG (pop1), START_MODEL ($T26/tel26_lp2th_th0_it60_k6.pt; zero = residuo nullo), CONFIG (d64th), NITER (15), POP_TS ("322 328 334"),
#            POP_NS (3), POP_SKIP_PS (500), POP_STRIDE (1), POP_LAMBDA (1), POP_HUBER (2), POP_DELTA (0.05),
#            START_F, START_U (stati k6 a 330 K chiuso e 345 K aperto), START_RE (stato a 300 K della catena th0),
#            MD_PS (50), SKIP_PS (5), RE_STEPS (200), RE_LR (2e-5), RE_ESS_MIN (0.5), CHECK (1), MAX_HOURS (23.3)
set -uo pipefail
A=${A:-/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold}
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=$R/tutorials/tel26
source $R/hpc/env_leonardo.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MLCG_PAINN_GRAPH=device

TAG=${TAG:-pop1}
START_MODEL=${START_MODEL:-$T26/tel26_lp2th_th0_it60_k6.pt}
NITER=${NITER:-15}
POP_TS=${POP_TS:-"322 328 334"}
POP_F_TS=${POP_F_TS:-}
POP_F_MAP=${POP_F_MAP:-f1}
POP_ORDER=${POP_ORDER:-tetrad}     # tract: U = bacino senza tratti G appaiati (c_a < POP_TRACT_CUT), 23/10
POP_TRACT_CUT=${POP_TRACT_CUT:-0.2}
# POP_MIN_FRAMES (23/10): una T entra nel vincolo solo se nei campioni dell'iterazione ci sono almeno
# POP_MIN_FRAMES frame nello stato e altrettanti fuori (etichette nette).  Senza frame di uno dei due
# stati la ripesatura non puo' spostare il loro peso relativo, e il gradiente delle code delle
# etichette morbide sposta il bacino campionato (pop1, pop3 it01-04: allentamento invece di fusione).
# Se nessuna T qualifica, l'iterazione non allena (modello copiato) e la MD prosegue.  0 = tutte le T.
POP_MIN_FRAMES=${POP_MIN_FRAMES:-0}
MD_TS=$(echo $POP_TS $POP_F_TS | tr ' ' '\n' | sort -n -u | tr '\n' ' ')
POP_NS=${POP_NS:-3}
POP_SKIP_PS=${POP_SKIP_PS:-500}
POP_STRIDE=${POP_STRIDE:-1}
POP_LAMBDA=${POP_LAMBDA:-1}
POP_HUBER=${POP_HUBER:-2}
POP_DELTA=${POP_DELTA:-0.05}
K6=$A/melt_painn/tel26_lp2th_th0_it60_k6
START_F=${START_F:-$K6/chiuso/T330.state.npz}
START_U=${START_U:-$K6/aperto/T345.state.npz}
START_RE=${START_RE:-$T26/state_tel26_lp2th_th0_it60.npz}
MD_PS=${MD_PS:-50}
SKIP_PS=${SKIP_PS:-5}
RE_STEPS=${RE_STEPS:-200}
RE_LR=${RE_LR:-2e-5}
RE_ESS_MIN=${RE_ESS_MIN:-0.5}
CHECK=${CHECK:-1}
MAX_HOURS=${MAX_HOURS:-23.3}
OUTDIR=${OUTDIR:-$A/relent_pop/$TAG}

# CONFIG (24/10): d64th = PaiNN con teste termiche (U dipende da T); d64 = PaiNN indipendente da T.
# Con le teste termiche e il vincolo solo ad alta T il residuo puo' restare stretto a 300 K e
# allentare F ad alta T (pop4: fusione graduale); senza, la cooperativita' resta quella dei prior.
CONFIG=${CONFIG:-$T26/tel26_training_config.d64th.json}
PRI_RE=$T26/cg_priors.lp2th.json
RB_RE=$T26/rigid_bodies_info.lp2th.json
DATA_RE=$T26/tel26_lp2th_dataset.bin
PRI_POP=$A/thermo_val/f/cg_priors.json
RB_POP=$A/cg/rigid_bodies_info.lp2_1c.t400.json
DATA_POP=$A/cg/tel26_lp2_1c_t400_dataset.bin
PYPRESSO=$R/espresso/build/pypresso
RELENT=$R/training/build/train_relent
# START_MODEL=zero: residuo nullo (train_relent --zero-init) con la CONFIG data: la prima MD riproduce i soli prior
START_FILES=("$START_MODEL" "$START_MODEL.manifest.json"); [[ $START_MODEL == zero ]] && START_FILES=()
for f in "${START_FILES[@]}" $CONFIG $PRI_RE $RB_RE $DATA_RE $PRI_POP $RB_POP $DATA_POP \
         "$START_F" "$START_U" "$START_RE" $PYPRESSO $RELENT $T26/tel26_lp2_dataset.bin; do
    [[ -e $f ]] || { echo "[ERROR] manca $f" >&2; exit 1; }
done
grep -aq -- "--pop-huber" $RELENT || { echo "[ERROR] train_relent non ha --pop: ricompilalo" >&2; exit 1; }
mkdir -p $OUTDIR && cd $OUTDIR || exit 1
exec 9> .lock
flock -n 9 || { echo "[ERROR] la catena $TAG e' gia' in corso in $OUTDIR" >&2; exit 1; }

DT=0.004
RE_MD_STEPS=$(python3 -c "print(int(round($MD_PS / $DT)))")
RE_LOG=25                                   # 0.1 ps
RE_SKIP=$(python3 -c "s=int(round($SKIP_PS / $DT)) // $RE_LOG * $RE_LOG; print(s)")
POP_STEPS=$(python3 -c "print(int(round($POP_NS * 1000 / $DT)))")
POP_LOG=1250                                # 5 ps
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); (( NGPU > 0 )) || NGPU=1
mapfile -t CPUS < <(python3 -c 'import os; print("\n".join(map(str, sorted(os.sched_getaffinity(0)))))')
T_START=$(date +%s)
echo "[relpop] $TAG in $OUTDIR: partenza $START_MODEL, T pop: U $POP_TS${POP_F_TS:+, F $POP_F_TS ($POP_F_MAP)}, $POP_NS ns per corsa, 300 K: $MD_PS ps"
echo "[relpop] T con almeno $POP_MIN_FRAMES frame per stato; lambda $POP_LAMBDA, Huber $POP_HUBER, delta $POP_DELTA, ordine $POP_ORDER (soglia per tratto $POP_TRACT_CUT), lr max $RE_LR, $NGPU GPU, ${#CPUS[@]} core"

both_states() {   # json di 19_pop_labels.py -> vero se ci sono >= POP_MIN_FRAMES frame dentro e fuori lo stato
    (( POP_MIN_FRAMES > 0 )) || return 0
    python3 -c "
import json, sys
s = json.load(open(sys.argv[1])); m = int(sys.argv[2])
n_in = sum(r['P_U_hard'] * r['frames'] for r in s['runs'] if r['frames'])
n_out = sum(r['frames'] for r in s['runs']) - n_in
sys.exit(0 if n_in >= m and n_out >= m else 1)" "$1" "$POP_MIN_FRAMES"
}
model_of() { printf "%s/%s_it%02d.pt" "$OUTDIR" "$TAG" "$1"; }
md_of()    { printf "%s/md_it%02d" "$OUTDIR" "$1"; }
manifest() { python3 $R/training/create_model_manifest.py --model "$1" --config $CONFIG --dataset $DATA_RE >/dev/null; }
next_lr()  { python3 -c "
import json, sys
cap = float(sys.argv[2])
try: lr = min(cap, 2.0 * float(json.load(open(sys.argv[1]))['lr_final']))
except Exception: lr = cap
print(f'{lr:.6g}')" "$1" "$RE_LR"; }

m0=$(model_of 0)
if [[ ! -s $m0 ]]; then
    if [[ $START_MODEL == zero ]]; then
        rm -f $m0.tmp.pt
        $RELENT --config $CONFIG --aa $DATA_RE --zero-init --steps 0 --out $m0.tmp.pt \
            --report ${m0%.pt}.relent.json > ${m0%.pt}.relent.log 2>&1 || { tail -5 ${m0%.pt}.relent.log; exit 1; }
        mv $m0.tmp.pt $m0
        echo "[relpop] modello di partenza nullo ($(basename $CONFIG)): $m0"
    else
        cp "$START_MODEL" $m0.tmp && mv $m0.tmp $m0
    fi
fi
[[ -s $m0.manifest.json ]] || manifest $m0

# MD col modello it(k) in md_it<k>: 300 K + coppie (T, lignaggio)
simulate() {
    local k=$1 D; D=$(md_of $k)
    [[ -e $D/done ]] && { echo "[relpop] MD di it$(printf %02d $k) gia' fatta"; return 0; }
    mkdir -p $D
    local P; P=$(md_of $((k - 1)))
    local i=0 pids=() name ck kt seed extra off
    # 300 K, 10 copie
    if [[ ! -s $D/re.samples.npz ]]; then
        if (( k == 0 )); then ck=$START_RE; else ck=$P/re.state.npz; fi
        mkdir -p $D/wd_re
        ( cd $D/wd_re && CUDA_VISIBLE_DEVICES=$(( i % NGPU )) exec taskset -c ${CPUS[$i]} \
            $PYPRESSO $R/simulation/run_cg_md.py --model $(model_of $k) --config $CONFIG \
            --priors $PRI_RE --rb_info $RB_RE --dataset $DATA_RE --checkpoint $ck --allow_checkpoint_mismatch \
            --steps $RE_MD_STEPS --dt $DT --kT 2.49 --gamma 20 --thermostat_seed $(( 1000 + k )) \
            --device cuda --neighbor_search verlet --log_interval $RE_LOG --energy_interval 10 --no_vtf \
            --energy_file $D/re.energy.csv --sample_npz $D/re.samples.npz --sample_start_step $RE_SKIP \
            --out_checkpoint $D/re.state.npz ) > $D/re.log 2>&1 &
        pids+=($!); i=$((i + 1))
    fi
    for T in $MD_TS; do
        kt=$(python3 -c "print(f'{0.008314462618*$T:.6f}')")
        for b in F U; do
            name=$D/${b}_T$T
            [[ -s $name.samples.npz ]] && continue
            off=0; [[ $b == U ]] && off=50000
            seed=$(( T * 7 + 13 + 100 * k + off ))
            extra=()
            # stato finale dell'iterazione precedente; per una T aggiunta a catena avviata
            # (o alla prima iterazione) si parte da START_F / START_U con velocita' nuove
            if (( k > 0 )) && [[ -s $P/${b}_T$T.state.npz ]]; then
                ck=$P/${b}_T$T.state.npz
            else
                if [[ $b == F ]]; then ck=$START_F; else ck=$START_U; fi
                extra=(--init_kT $kt --velocity_seed $(( seed + 7919 )))
            fi
            mkdir -p $D/wd_${b}_T$T
            ( cd $D/wd_${b}_T$T && CUDA_VISIBLE_DEVICES=$(( i % NGPU )) exec taskset -c ${CPUS[$i]} \
                $PYPRESSO $R/simulation/run_cg_md.py --model $(model_of $k) --config $CONFIG \
                --priors $PRI_POP --rb_info $RB_POP --dataset $DATA_POP --checkpoint $ck --allow_checkpoint_mismatch \
                --steps $POP_STEPS --dt $DT --gamma 2 --kT $kt --thermostat_seed $seed ${extra[@]+"${extra[@]}"} \
                --device cuda --neighbor_search verlet --log_interval $POP_LOG --energy_interval 10 --no_vtf \
                --energy_file $name.energy.csv --sample_npz $name.samples.npz --out_checkpoint $name.state.npz ) \
                > $name.log 2>&1 &
            pids+=($!); i=$((i + 1))
        done
    done
    echo "[relpop] $(date '+%F %T') MD di it$(printf %02d $k): ${#pids[@]} corse"
    local fail=0
    for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
    (( fail == 0 )) || { echo "[ERROR] $fail corse fallite in $D (vedi *.log)" >&2; return 1; }
    for f in $D/re.samples.npz $(for T in $MD_TS; do echo $D/F_T$T.samples.npz $D/U_T$T.samples.npz; done); do
        [[ -s $f ]] || { echo "[ERROR] manca $f" >&2; return 1; }
    done
    touch $D/done
}

cur_lr=$RE_LR
last_dur=0
for n in $(seq 1 $NITER); do
    prev=$((n - 1))
    mn=$(model_of $n)
    if [[ -s $mn && -s $mn.manifest.json ]]; then
        [[ -s ${mn%.pt}.relent.json ]] && cur_lr=$(next_lr ${mn%.pt}.relent.json)
        continue
    fi
    now=$(date +%s)
    if (( last_dur > 0 )) && python3 -c "import sys; sys.exit(0 if $now + 1.15 * $last_dur > $T_START + $MAX_HOURS * 3600 else 1)"; then
        echo "[relpop] l'iterazione $n non ci sta nel job (ultima: $((last_dur / 60)) min): risottometti lo script"
        break
    fi
    t0=$now
    simulate $prev || exit 1
    D=$(md_of $prev)
    [[ -s $D/re.bin ]] || python3 $R/preprocessing/samples_to_cg_bin.py --template $DATA_RE \
        --samples $D/re.samples.npz --out $D/re.bin > /dev/null || exit 1
    specs=()
    for T in $POP_TS; do
        if [[ ! -s $D/pop_T$T.labels ]]; then
            rm -f $D/pop_T$T.bin
            python3 $U/19_pop_labels.py --T $T --template $DATA_POP --samples $D/F_T$T.samples.npz $D/U_T$T.samples.npz \
                --skip-ps $POP_SKIP_PS --stride $POP_STRIDE --delta $POP_DELTA --order $POP_ORDER --tract-cut $POP_TRACT_CUT \
                --out-prefix $D/pop_T$T > $D/pop_T$T.txt \
                || { cat $D/pop_T$T.txt; exit 1; }
        fi
        grep -v POPSPEC $D/pop_T$T.txt
        if both_states $D/pop_T$T.json; then
            specs+=(--pop "$(grep '^\[POPSPEC\]' $D/pop_T$T.txt | awk '{print $2}'),$POP_LAMBDA")
        else
            echo "  T $T: esclusa (meno di $POP_MIN_FRAMES frame in uno dei due stati)"
        fi
    done
    for T in $POP_F_TS; do
        if [[ ! -s $D/popF_T$T.labels ]]; then
            rm -f $D/popF_T$T.bin
            python3 $U/19_pop_labels.py --T $T --state F --map $POP_F_MAP --template $DATA_POP \
                --samples $D/F_T$T.samples.npz $D/U_T$T.samples.npz --skip-ps $POP_SKIP_PS --stride $POP_STRIDE \
                --delta $POP_DELTA --out-prefix $D/popF_T$T > $D/popF_T$T.txt || { cat $D/popF_T$T.txt; exit 1; }
        fi
        grep -v POPSPEC $D/popF_T$T.txt
        if both_states $D/popF_T$T.json; then
            specs+=(--pop "$(grep '^\[POPSPEC\]' $D/popF_T$T.txt | awk '{print $2}'),$POP_LAMBDA")
        else
            echo "  T $T (F): esclusa (meno di $POP_MIN_FRAMES frame in uno dei due stati)"
        fi
    done
    if (( ${#specs[@]} == 0 )); then
        echo "[relpop] $(date '+%F %T') it$(printf %02d $n): nessuna T con entrambi gli stati campionati, modello copiato senza allenamento"
        cp $(model_of $prev) $mn.tmp.pt && mv $mn.tmp.pt $mn
        manifest $mn
        echo '{"skipped": true}' > ${mn%.pt}.relent.json
        echo "[POP-ITER] it$(printf %02d $n) saltata (nessuna T con entrambi gli stati)"
        last_dur=$(( $(date +%s) - t0 ))
        continue
    fi
    common=(--config $CONFIG --aa $DATA_RE --cg $D/re.bin "${specs[@]}" --pop-huber $POP_HUBER
            --residues-per-copy 26 --batch-aa 8 --batch-cg 8 --batch-pop 8)
    if [[ "$CHECK" == 1 && ! -e $OUTDIR/check.ok ]]; then
        echo "[relpop] verifica del gradiente (CPU, float64)"
        CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $RELENT "${common[@]}" --in $(model_of $prev) --check-gradient > $OUTDIR/check.log 2>&1
        grep -E "^\[(CHECK|OK|FAIL)\]" $OUTDIR/check.log | grep -E "migliore|miglior|OK|FAIL|popolazioni|batching"
        grep -q "^\[OK\]" $OUTDIR/check.log || { echo "[ERROR] verifica del gradiente fallita: $OUTDIR/check.log" >&2; exit 1; }
        touch $OUTDIR/check.ok
    fi
    rep=${mn%.pt}.relent.json
    echo "[relpop] $(date '+%F %T') train_relent: it$(printf %02d $prev) -> it$(printf %02d $n) (lr $cur_lr)"
    rm -f $mn.tmp.pt
    CUDA_VISIBLE_DEVICES=0 $RELENT "${common[@]}" --in $(model_of $prev) --out $mn.tmp.pt --report $rep \
        --steps $RE_STEPS --lr $cur_lr --ess-min $RE_ESS_MIN --holdout-frac 0.2 --eval-every 10 \
        --max-backtracks 6 --seed $((42 + n)) --device cuda > ${mn%.pt}.relent.log 2>&1 \
        || { tail -20 ${mn%.pt}.relent.log; exit 1; }
    mv $mn.tmp.pt $mn
    manifest $mn
    cur_lr=$(next_lr $rep)
    python3 - "$rep" "$n" <<'PYEOF'
import json, math, sys
r = json.load(open(sys.argv[1]))
sig = lambda x: 1.0 / (1.0 + math.exp(-x))
st = ["F" if "/popF_" in q["labels"] else "U" for q in r.get("pop", [])]
pops = "  ".join(f"{p['T_K']:g} K: P_{s} {sig(p['L0']):.3f} -> {sig(p['L']):.3f} (esp {sig(p['target_L']):.3f})"
                 for s, p in zip(st, r.get("pop_final", [])))
print(f"[POP-ITER] it{int(sys.argv[2]):02d} passo {r['best_step']} punteggio {r['best_dS']:.3f} "
      f"arresto {r['stop_reason']} ESS {r['ess_last']:.3f} lr {r['lr_final']:.3g} ({round(r['seconds'])} s)  {pops}")
PYEOF
    last_dur=$(( $(date +%s) - t0 ))
done

echo; echo "[relpop] riepilogo (popolazione del modello sui campioni di MD dell'iterazione, per lignaggio):"
for D in $OUTDIR/md_it*; do
    [[ -e $D/done ]] || continue
    for T in $MD_TS; do for pre in pop popF; do
        [[ -s $D/${pre}_T$T.json ]] || continue
        python3 -c "
import json, sys
s = json.load(open(sys.argv[1]))
st = s.get('state', 'U')
print(f\"  {sys.argv[2]}  T {s['T_K']:g}  P_{st}  \" + '  '.join(f\"{r['samples'].split('/')[-1].split('.')[0]} {r['P_U_hard']:.2f}\" for r in s['runs'])
      + f\"   esp {s['P_U_exp']:.2f}\")" $D/${pre}_T$T.json $(basename $D)
    done; done
done
