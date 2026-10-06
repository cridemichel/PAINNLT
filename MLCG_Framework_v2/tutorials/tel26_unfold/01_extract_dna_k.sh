#!/bin/bash
# Estrae DNA + K+ (molecole intere, ogni 10 ps) dalle corse AA a 400 K di Giulia.
#
# PERCHE'
#     Le sue traiettorie solo DNA non contengono gli ioni: per sapere se e quando
#     i K+ entrano o escono dal canale e quando la tetrade 3 (G6, G10, G18, G24)
#     si apre servono DNA e K+ insieme, anche per l'equilibratura (20 ns), dove
#     la tetrade 3 risulta gia' aperta alla fine.
#
# USO (Leonardo, come job: legge ~240 GB di xtc)
#     A=/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold
#     U=/leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/MLCG_Framework_v2/tutorials/tel26_unfold
#     sbatch -A IscrB_G4MES_0 -p dcgp_usr_prod --cpus-per-task=4 --mem=8G --time=06:00:00 \
#         -J aa_dnak -o /leonardo_work/IscrB_G4MES/cdemiche/PAINNLT/logs/slurm-aa_dnak-%j.out \
#         --wrap "bash $U/01_extract_dna_k.sh"
#     Uscite in $A/dnak/: dnak.ndx, equil_dna_k.{gro,xtc}, prod-{1,2,3}_dna_k.xtc
set -euo pipefail
G=${G:-/leonardo_work/IscrB_G4MES/GiuliaT/AllAtom}
A=${A:-/leonardo_work/IscrB_G4MES/cdemiche/AA_unfold}
DT=${DT:-10}
module load profile/chem-phys >/dev/null 2>&1 || true
module load gromacs >/dev/null 2>&1 || true
GMX=$(command -v gmx || command -v gmx_mpi)
echo "[INFO] gmx: $GMX"
mkdir -p $A/dnak && cd $A/dnak

$GMX select -s $G/equil_hybrid_tel26.tpr -select 'resname DT5 DT DA DG DT3 K' -on dnak.ndx
# equilibratura: struttura finale e traiettoria
echo 0 | $GMX trjconv -s $G/equil_hybrid_tel26.tpr -f $G/equil_hybrid_tel26.gro -n dnak.ndx -o equil_dna_k.gro -pbc mol
echo 0 | $GMX trjconv -s $G/equil_hybrid_tel26.tpr -f $G/equil_hybrid_tel26.xtc -n dnak.ndx -o equil_dna_k.xtc -pbc mol -dt $DT
# produzione, parte per parte
for k in 1 2 3; do
    f=$G/prod-$k.part000$k.xtc
    [ -f $f ] || { echo "[WARN] manca $f"; continue; }
    echo 0 | $GMX trjconv -s $G/md_hybrid_tel26.tpr -f $f -n dnak.ndx -o prod-${k}_dna_k.xtc -pbc mol -dt $DT
done
ls -la $A/dnak
echo "[DONE]"
