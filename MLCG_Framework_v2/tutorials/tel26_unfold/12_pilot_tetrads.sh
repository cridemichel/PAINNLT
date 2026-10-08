#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p dcgp_usr_prod
#SBATCH -A IscrB_G4MES_0
#SBATCH -t 01:30:00
#SBATCH -J pil_tet
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-pil_tet-%j.out
# Tetradi e K+ lungo le repliche del pilota AA senza K+ nel canale (04_run_pilot.sh).
#
# Per ogni replica: tpr ridotto al gruppo DNA_K (convert-tpr), traiettoria con le molecole
# intere (trjconv -pbc mol) e 02_tetrads_k.py con la struttura di partenza ridotta a DNA_K
# (em.gro) come topologia: stessi atomi e stessi nomi della corsa.  La stessa struttura fa da
# "equilibratura" di un frame (non ricomposta: quella riga del riassunto va ignorata), cosi' il
# tempo della produzione parte da 0.
# Uscite in $P/ana: rep<r>_dnak.xtc, tetrads_rep<r>.{npz,png} e il riassunto nel log.
#
# USO (Leonardo):  sbatch $U/12_pilot_tetrads.sh      (si puo' rilanciare: rifa' tutto da capo)
# Variabili: P (cartella del pilota), NREP (4), STRIDE (1 = un frame ogni 10 ps), ION (K o LI)
A=${A:-/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold}
P=${P:-$A/pilot_noK400}
NREP=${NREP:-4}
STRIDE=${STRIDE:-1}
ION=${ION:-K}         # LI per il pilota con Li+ (03 --cation Li)
R=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2
U=$R/tutorials/tel26_unfold
mkdir -p $P/ana

( module purge
  module load profile/chem-phys fftw/3.3.10--gcc--12.2.0 openblas/0.3.24--gcc--12.2.0 gromacs/2023.3--gcc--12.2.0-cuda-12.1
  set -e
  cd $P
  echo DNA_K | gmx convert-tpr -s rep1.tpr -n index.ndx -o ana/dnak.tpr
  echo DNA_K | gmx trjconv -s rep1.tpr -f em.gro -n index.ndx -o ana/start_dnak.gro
  for r in $(seq 1 $NREP); do
      xtc=$(ls rep$r/rep$r*.xtc | head -1)
      echo "[pil_tet] rep$r: $xtc"
      echo System | gmx trjconv -s ana/dnak.tpr -f $xtc -pbc mol -o ana/rep${r}_dnak.xtc
  done ) || { echo "[ERROR] parte GROMACS fallita" >&2; exit 1; }

source $R/hpc/env_leonardo.sh
cd $P/ana
for r in $(seq 1 $NREP); do
    echo; echo "=== rep$r"
    python3 $U/02_tetrads_k.py start_dnak.gro start_dnak.gro rep${r}_dnak.xtc \
        --stride $STRIDE --ion-resname $ION --out tetrads_rep$r
done
