#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -p boost_usr_prod
#SBATCH -A IscrB_G4MES
#SBATCH -t 02:00:00
#SBATCH -J bondpar
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-bondpar-%j.out
# Contatti Morse come legami reversibili (run_cg_md --pair_specific_morse_runtime bonded-switched)
# contro i marcatori non legati di produzione (marker-nonbonded): devono dare la stessa fisica.
#
# PERCHE' (2/11): con i marcatori ogni estremo di contatto ha un tipo di particella proprio, per
# copia; ESPResSo tiene parametri per ogni coppia di tipi, quindi la memoria cresce come il
# quadrato delle copie (centinaia di TEL26 per la S(q) totale: impossibile).  Un legame agisce su
# due particelle date e non consuma tipi.  MorseBond con switch_start >= 0 (morse_bond.hpp) e' la
# stessa funzione del Morse non legato con raccordo, inerte (non "rotto") oltre r_cut.
#
# PROVE (una GPU), per ciascun sistema: forze e coppie di ogni corpo rigido sulla configurazione del
# checkpoint (--dump_initial_forces) ed energia al passo 0, con i due runtime; poi dinamica breve
# col runtime a legami dallo stato aperto (contatti oltre r_cut: non deve fermarsi) e ms/passo.
#   f1F   una copia, prior termici f a 318 K, stato F (md_it21/F_T318 di pop6)
#   f1U   una copia, prior f a 345 K, stato aperto (thermo_val/f/aperto/T345_r0)
#   lp10  10 copie, prior lp2th a 300 K (md_it21/re.state.npz di pop6)
# Esito: [PARITY] per sistema con le differenze massime; [OK]/[FAIL] finali.
#
# USO (Leonardo, dopo l'installatore e il build di ESPResSo):  sbatch $U/24_bonded_parity.sh
# Variabili: MODEL ($A/relent_pop/pop6/pop6_it21.pt), STEPS (5000), OUT ($A/bonded_parity)
set -uo pipefail
A=${A:-/leonardo_work/IscrC_G4UNFOLD/cdemiche/AA_unfold}
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
T26=$R/tutorials/tel26
source $R/hpc/env_leonardo.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MLCG_PAINN_GRAPH=device
MODEL=${MODEL:-$A/relent_pop/pop6/pop6_it21.pt}
STEPS=${STEPS:-5000}
OUT=${OUT:-$A/bonded_parity}
PYPRESSO=$R/espresso/build/pypresso
CONFIG=$T26/$(python3 -c 'import json,sys,os;print(os.path.basename(json.load(open(sys.argv[1]))["config_path"]))' "$MODEL.manifest.json") || exit 1
$PYPRESSO -c "import espressomd.interactions as i; b = i.MorseBond(D=1.0, a=3.0, r_0=0.3, r_cut=15.0, switch_start=0.9); assert b.switch_start == 0.9" \
    || { echo "[ERROR] ESPResSo senza MorseBond(switch_start): installatore + build" >&2; exit 1; }
mkdir -p $OUT && cd $OUT || exit 1

# nome | prior | rb_info | dataset | checkpoint | T (K)
SYSTEMS=(
  "f1F|$A/thermo_val/f/cg_priors.json|$A/cg/rigid_bodies_info.lp2_1c.t400.json|$A/cg/tel26_lp2_1c_t400_dataset.bin|$A/relent_pop/pop6/md_it21/F_T318.state.npz|318"
  "f1U|$A/thermo_val/f/cg_priors.json|$A/cg/rigid_bodies_info.lp2_1c.t400.json|$A/cg/tel26_lp2_1c_t400_dataset.bin|$A/thermo_val/f/aperto/T345_r0.state.npz|345"
  "lp10|$T26/cg_priors.lp2th.json|$T26/rigid_bodies_info.lp2th.json|$T26/tel26_lp2th_dataset.bin|$A/relent_pop/pop6/md_it21/re.state.npz|299.5"
)
fail=0
for item in "${SYSTEMS[@]}"; do
    IFS='|' read -r name pri rb ds ck T <<< "$item"
    for f in $pri $rb $ds $ck; do [[ -s $f ]] || { echo "[ERROR] manca $f" >&2; exit 1; }; done
    kt=$(python3 -c "print(f'{0.008314462618*$T:.6f}')")
    common=(--model $MODEL --config $CONFIG --priors $pri --rb_info $rb --dataset $ds --checkpoint $ck
            --allow_checkpoint_mismatch --dt 0.004 --gamma 2 --kT $kt --device cuda --neighbor_search verlet --no_vtf)
    for rt in marker-nonbonded bonded-switched; do
        $PYPRESSO $R/simulation/run_cg_md.py "${common[@]}" --pair_specific_morse_runtime $rt \
            --dump_initial_forces $name.$rt.forces.npz > $name.$rt.dump.log 2>&1 || { tail -5 $name.$rt.dump.log; fail=1; continue; }
        $PYPRESSO $R/simulation/run_cg_md.py "${common[@]}" --pair_specific_morse_runtime $rt \
            --steps 0 --energy_file $name.$rt.e0.csv > $name.$rt.e0.log 2>&1 || { tail -5 $name.$rt.e0.log; fail=1; }
    done
    python3 - "$name" <<'PYEOF' || fail=1
import csv, sys
import numpy as np
n = sys.argv[1]
a = np.load(f"{n}.marker-nonbonded.forces.npz"); b = np.load(f"{n}.bonded-switched.forces.npz")
assert np.array_equal(a["mol"], b["mol"]), "molecole diverse"
dpos = np.abs(a["com"] - b["com"]).max()
fs = np.abs(a["force"]).max(); ts = np.abs(a["torque_lab"]).max()
df = np.abs(a["force"] - b["force"]).max(); dt = np.abs(a["torque_lab"] - b["torque_lab"]).max()
def e0(rt):
    row = next(csv.DictReader(open(f"{n}.{rt}.e0.csv")))
    return float(row["E_tot"]) - float(row["E_kin"]), float(row["E_ml"])
(ua, mla), (ub, mlb) = e0("marker-nonbonded"), e0("bonded-switched")
ok = dpos < 1e-9 and df <= 1e-5 * max(fs, 1.0) and dt <= 1e-5 * max(ts, 1.0) and abs(ua - ub) <= 1e-6 * max(abs(ua), 1.0)
print(f"[PARITY] {n}: |dCOM| {dpos:.2e} nm, |dF| max {df:.3e} (|F| max {fs:.3e}), |dtau| max {dt:.3e} "
      f"(|tau| max {ts:.3e}) kJ/mol/nm; U pot {ua:.6f} / {ub:.6f} (diff {ua - ub:.2e}), E_ml {mla:.6f} / {mlb:.6f}  "
      + ("OK" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PYEOF
done

# dinamica dallo stato aperto: contatti oltre r_cut, il legame non deve "rompersi"; ms/passo per runtime
for rt in marker-nonbonded bonded-switched; do
    IFS='|' read -r name pri rb ds ck T <<< "${SYSTEMS[1]}"
    kt=$(python3 -c "print(f'{0.008314462618*$T:.6f}')")
    t0=$(date +%s.%N)
    $PYPRESSO $R/simulation/run_cg_md.py --model $MODEL --config $CONFIG --priors $pri --rb_info $rb --dataset $ds \
        --checkpoint $ck --allow_checkpoint_mismatch --dt 0.004 --gamma 2 --kT $kt --device cuda --neighbor_search verlet \
        --no_vtf --steps $STEPS --log_interval 250 --energy_interval 10 --energy_file dyn.$rt.csv \
        --pair_specific_morse_runtime $rt > dyn.$rt.log 2>&1 || { echo "[FAIL] dinamica $rt:"; tail -5 dyn.$rt.log; fail=1; continue; }
    t1=$(date +%s.%N)
    python3 -c "print(f'[DYN] $rt: $STEPS passi in {($t1 - $t0):.0f} s (avvio compreso) = {1000 * ($t1 - $t0) / $STEPS:.2f} ms/passo')"
done
(( fail == 0 )) && echo "[OK] runtime a legami equivalente ai marcatori" || echo "[FAIL] vedi i log in $OUT"
