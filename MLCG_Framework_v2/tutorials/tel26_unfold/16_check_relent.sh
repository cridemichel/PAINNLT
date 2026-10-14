#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -p boost_usr_prod
#SBATCH -A IscrB_G4MES
#SBATCH -t 04:00:00
#SBATCH -J chk_relent
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-chk_relent-%j.out
# Controllo intermedio di una catena di entropia relativa sui prior termici (06_relent.sh,
# PRIOR_SET=lp2th): la struttura a 300 K torna quella AA?
#
# Due produzioni a 10 copie sulla stessa GPU, in parallelo, dallo stesso stato:
#   <modello>   PaiNN attivo sopra lp2th (D(T) risolto a 299,5 K da run_cg_md)
#   pri_lp2th   soli prior lp2th (DISABLE_ML=1): quanto aggiunge PaiNN (si salta se esiste gia')
# Poi, contro il riferimento AA (tel26_lp2_dataset.bin), scartando SKIP_PS iniziali:
#   53_tel26_report_data.py   sovrapposizioni P(r) intra / g(r) inter per canale
#                             (tutti i siti, B3-B3, B5-B5, S-S) e il tetto AA meta' contro meta'
#   46_tel26_structure.py     rmsd_core, rmsd_loops, Q, Rg (sovrapposizioni e JSD)
#   08_tetrad_melt.py         tetradi intatte con q mediato su 1 ns (AA, PaiNN, soli prior)
# Riferimento (report 8/10, lp2 + re1_it30): tutti i siti 0,9945, B3-B3 0,976, B5-B5 0,967,
# S-S 0,958; soli prior lp2 0,980 / 0,841 / 0,916 / 0,924; tetto 0,9955 / 0,979 / 0,971 / 0,961.
#
# USO (Leonardo):  sbatch $U/16_check_relent.sh
# Variabili: MODEL (tel26_lp2th_th0_it30.pt), CHECKPOINT (state_<catena>_it30.npz), PRIOR_SET (lp2th),
#            NS (2 ns per produzione), SKIP_PS (200), OUT (tutorials/tel26/check_relent)
set -uo pipefail
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=$R/tutorials/tel26
source $R/hpc/env_leonardo.sh
cd $T26 || exit 1
export PRIOR_SET=${PRIOR_SET:-lp2th}
MODEL=${MODEL:-tel26_lp2th_th0_it30.pt}
TAG=$(basename "$MODEL" .pt)
CHECKPOINT=${CHECKPOINT:-state_${TAG}.npz}
NS=${NS:-2}
SKIP_PS=${SKIP_PS:-200}
OUT=${OUT:-$T26/check_relent}
REF=tel26_lp2_dataset.bin
mkdir -p "$OUT"
for f in "$MODEL" "$CHECKPOINT" "$REF"; do [[ -s $f ]] || { echo "[ERROR] manca $T26/$f" >&2; exit 1; }; done
STEPS=$(python3 -c "print(int(round($NS * 1000 / 0.004)))")
T_END=$(python3 -c "print(int($NS * 1000))")
echo "[chk] $MODEL da $CHECKPOINT, PRIOR_SET $PRIOR_SET, $NS ns ($STEPS passi), uscite in $OUT"

run() {   # nome disable_ml
    local name=$1
    [[ -s $OUT/$name.samples.npz ]] && { echo "[chk] $name gia' fatto, salto"; return 0; }
    env MODEL="$MODEL" CHECKPOINT="$CHECKPOINT" DEVICE=cuda CG_STEPS=$STEPS LOG_INTERVAL=125 \
        ${2:+DISABLE_ML=1} SAMPLE_NPZ="$OUT/$name.samples.npz" \
        MD_EXTRA_ARGS="--allow_checkpoint_mismatch --no_vtf --energy_file $OUT/$name.energy.csv" \
        bash $T26/05_run_espresso.sh > "$OUT/$name.log" 2>&1 \
        || { echo "[chk] $name FALLITO: vedi $OUT/$name.log" >&2; return 1; }
    grep -m1 "prior termici" "$OUT/$name.log" || echo "[WARN] $name: nessuna riga 'prior termici' nel log"
}
run "$TAG" "" & p1=$!
run pri_lp2th 1 & p2=$!
wait $p1; s1=$?; wait $p2; s2=$?
(( s1 == 0 )) || exit 1
runs=(--run "$TAG=$OUT/$TAG.samples.npz")
specs=("$TAG=$OUT/$TAG.samples.npz#$SKIP_PS:$T_END")
tet=("$TAG=$OUT/$TAG.samples.npz")
if (( s2 == 0 )); then
    runs+=(--run "pri_lp2th=$OUT/pri_lp2th.samples.npz")
    specs+=("pri_lp2th=$OUT/pri_lp2th.samples.npz#$SKIP_PS:$T_END")
    tet+=("pri_lp2th=$OUT/pri_lp2th.samples.npz")
fi

echo; echo "=== sovrapposizioni P(r) / g(r) per canale (53)"
python3 $T26/53_tel26_report_data.py $REF "${runs[@]}" --skip-ps $SKIP_PS --out "$OUT/report_data_$TAG.npz" 2>&1 | tail -n 8
echo; echo "=== struttura della copia (46)"
python3 $T26/46_tel26_structure.py $REF half2=@ref:0.5:1 "${specs[@]}" --ref-range 0:0.5 2>&1 | tail -n 25
echo; echo "=== tetradi, q mediato su 1 ns (08)"
python3 $U/08_tetrad_melt.py aa=@bin:$T26/$REF "${tet[@]}" --smooth-ns 1 --window 0.1:1 \
    --json "$OUT/tetrads_$TAG.json" --plot "$OUT/tetrads_$TAG" 2>&1 | tail -n 8
