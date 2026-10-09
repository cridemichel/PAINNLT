#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrB_G4MES
#SBATCH -t 24:00:00
#SBATCH -J aa_pilot
#SBATCH -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-aa_pilot-%j.out
# Pilota AA: NREP copie del TEL26 a 400 K senza K+ nel canale, una per GPU sullo stesso nodo.
# Con 03 --exclude-r il canale resta vietato ai cationi per tutta la corsa (restraint di tipo 10
# catione-O6 in [ intermolecular_interactions ]: calcolati su CPU se -bonded gpu non li supporta).
#
# Alla prima esecuzione (nessun rep1.tpr): grompp + minimizzazione (em) + grompp delle copie.
# Poi mdrun in parallelo, 8 core e 1 GPU per copia; ogni esecuzione successiva riparte dai
# checkpoint (-cpi).  Per proseguire oltre le 24 h si risottomette lo stesso script.
#
# USO (Leonardo, dopo 03_make_pilot_inputs.py)
#     sbatch $U/04_run_pilot.sh
#     sbatch --dependency=afterany:<jobid> $U/04_run_pilot.sh     # continuazione
# Variabili: P (cartella del pilota), NREP (4), MAXH (23.5)
P=${P:-/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold/pilot_noK400}
NREP=${NREP:-4}
MAXH=${MAXH:-23.5}
TOP=ibrido_noions_spce.top

# ambiente pulito: il job eredita i moduli della shell da cui si lancia sbatch (es. il venv PaiNN, CUDA 12.2/12.6)
module purge
module load profile/chem-phys
module load fftw/3.3.10--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load gromacs/2023.3--gcc--12.2.0-cuda-12.1
set -euo pipefail     # dopo i module load (Lmod non regge set -u)
export GMX_GPU_DD_COMMS=true GMX_GPU_PME_PP_COMMS=true GMX_FORCE_UPDATE_DEFAULT_GPU=true
cd $P
echo "[INFO] $(date)  cartella $P  copie $NREP  maxh $MAXH"
# se un mdp attiva il pull code, l'update su GPU non e' garantito: lo si lascia scegliere a mdrun.
if grep -qiE '^[[:space:]]*pull[[:space:]]*=[[:space:]]*yes' md_rep1.mdp; then
    unset GMX_FORCE_UPDATE_DEFAULT_GPU
    echo "[INFO] pull attivo ($(grep -iE '^[[:space:]]*pull-ncoords' md_rep1.mdp | awk '{print $3}') coordinate): update scelto da mdrun"
fi

if [ ! -f rep1.tpr ]; then
    gmx grompp -f em.mdp -c start.gro -p $TOP -n index.ndx -o em.tpr
    gmx mdrun -deffnm em -ntmpi 1 -ntomp 8 -nb gpu
    for r in $(seq 1 $NREP); do
        gmx grompp -f md_rep$r.mdp -c em.gro -p $TOP -n index.ndx -o rep$r.tpr
    done
    rm -f \#*
fi

for r in $(seq 1 $NREP); do
    mkdir -p rep$r
    ( cd rep$r && export OMP_NUM_THREADS=8 && \
      gmx mdrun -s ../rep$r.tpr -deffnm rep$r -cpi rep$r.cpt -ntmpi 1 -ntomp 8 \
          -nb gpu -pme gpu -bonded gpu -gpu_id $((r - 1)) \
          -pin on -pinoffset $(( (r - 1) * 8 )) -pinstride 1 -maxh $MAXH \
          > mdrun.out 2>&1 ) &
done
wait
for r in $(seq 1 $NREP); do
    echo "=== rep$r"; grep -E '^Performance:' rep$r/rep$r.log | tail -1 || true
    tail -n 2 rep$r/mdrun.out
done
echo "[DONE] $(date)"
