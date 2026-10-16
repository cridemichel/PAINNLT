#!/bin/bash
# Scansione del fattore entropico di PaiNN: U_ML(T) = U_ML(T0) [1 - KAPPA (T - T0)/T0], T0 = 300 K.
#
# PERCHE' (15/10): con il residuo th0_it60 allenato a 300 K sopra i prior termici il TEL26 non
# fonde piu' fino a 380 K (17_melt_painn.sh): PaiNN da solo tiene il quadruplex, perche'
# l'entropia relativa a 300 K non vincola la stabilita' di F rispetto a U.  Una testa entropica
# proporzionale a quella entalpica (train_relent --thermo-scale) lascia identico il modello a
# 300 K e lo indebolisce linearmente con T: un solo parametro, da scegliere sulla Tm come per i
# prior (09/13).  Il fattore si annulla a T = T0 (1 + 1/KAPPA) e oltre cambia segno: le T della
# scansione devono restare sotto.
#
# COSA FA (Leonardo)
#   1. per ogni KAPPA (su un nodo con GPU: il modello di partenza ha tensori CUDA e sul login
#      torch::load fallisce): modello termico tel26_lp2th_th0_it60_k<KAPPA>.pt (train_relent
#      --steps 0 --thermo-scale, config d64th) e il suo manifest;
#   2. se SUBMIT=1, un job 17_melt_painn.sh per KAPPA su 1 GPU + 8 core (6 corse: TS x chiuso/aperto),
#      con A nell'area dati indicata.
#
# USO:  sbatch -p boost_usr_prod -A IscrB_G4MES --gres=gpu:1 --cpus-per-task=8 -t 00:20:00 \
#           --wrap "source $R/hpc/env_leonardo.sh && bash $U/18_kappa_scan.sh"   (solo modelli)
#       SUBMIT=1 bash $U/18_kappa_scan.sh             (modelli + job)
# Variabili: KAPPAS ("2 4 6"), BASE (tel26_lp2th_th0_it60), TS ("315 330 345"), CG_STEPS (6250000),
#            A (/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold), SUBMIT
set -euo pipefail
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=$R/tutorials/tel26
KAPPAS=${KAPPAS:-"2 4 6"}
BASE=${BASE:-tel26_lp2th_th0_it60}
TS=${TS:-"315 330 345"}
CG_STEPS=${CG_STEPS:-6250000}
export A=${A:-/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold}
CONFIG=tel26_training_config.d64th.json
RELENT=$R/training/build/train_relent
cd $T26
for f in $BASE.pt $CONFIG tel26_lp2th_dataset.bin $RELENT; do [[ -e $f ]] || { echo "[ERROR] manca $f" >&2; exit 1; }; done
# la stringa dell'opzione e' nel binario: niente esecuzione (sul login, senza GPU, train_relent
# puo' bloccarsi nell'inizializzazione di CUDA)
grep -aq -- "--thermo-scale" "$RELENT" || { echo "[ERROR] train_relent non ha --thermo-scale: ricompilalo" >&2; exit 1; }
for k in $KAPPAS; do
    tag=${BASE}_k$k
    for T in $TS; do
        python3 -c "import sys; k,T=float(sys.argv[1]),float(sys.argv[2]); sys.exit(0 if k*(T-300)/300 < 1 else 1)" $k $T \
            || { echo "[ERROR] KAPPA $k: a $T K il fattore e' <= 0" >&2; exit 1; }
    done
    if [[ ! -s $tag.pt ]]; then
        $RELENT --config $CONFIG --aa tel26_lp2th_dataset.bin --in $BASE.pt --out $tag.pt \
            --steps 0 --thermo-scale $k --device cpu --report $tag.make.json > $tag.make.log 2>&1 \
            || { echo "[ERROR] train_relent per KAPPA $k: vedi $T26/$tag.make.log" >&2; tail -5 $tag.make.log; exit 1; }
        grep -E "thermo-scale|modello" $tag.make.log
        python3 $R/training/create_model_manifest.py --model $tag.pt --config $CONFIG --dataset tel26_lp2th_dataset.bin
    fi
    ls -l $tag.pt $tag.pt.manifest.json
    if [[ -n "${SUBMIT:-}" ]]; then
        sbatch -J kap$k --gres=gpu:1 --cpus-per-task=8 --mem=60G -t 24:00:00 \
            --export=ALL,A=$A,MODEL=$T26/$tag.pt,TS="$TS",CG_STEPS=$CG_STEPS $U/17_melt_painn.sh
    fi
done
