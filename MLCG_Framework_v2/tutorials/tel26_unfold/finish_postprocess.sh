#!/bin/bash
#SBATCH -N1
#SBATCH --job-name=finish_post
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -t 03:00:00
#SBATCH -A IscrB_G4MES

module load profile/chem-phys
module load fftw/3.3.10--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load gromacs/2023.3--gcc--12.2.0-cuda-12.1

export OMP_NUM_THREADS=8

# Rimuoviamo il file parziale incompleto
rm -f trj_tel26_dna_pbc.xtc trj_tel26_dna_fit.xtc rmsd_dna.xvg

echo "=== 1. Ricentratura PBC ed estrazione DNA (salvando ogni 10 ps) ==="
echo -e "1\n1" | gmx trjconv -s md_hybrid_tel26.tpr -f trj_tel26_raw.xtc -o trj_tel26_dna_pbc.xtc -pbc mol -center -dt 10

echo "=== 2. Fit rototraslazionale per VMD ==="
echo -e "1\n1" | gmx trjconv -s md_hybrid_tel26.tpr -f trj_tel26_dna_pbc.xtc -o trj_tel26_dna_fit.xtc -fit rot+trans

echo "=== 3. Calcolo RMSD completo (750 ns) ==="
echo -e "1\n1" | gmx rms -s md_hybrid_tel26.tpr -f trj_tel26_dna_pbc.xtc -o rmsd_dna.xvg -tu ns

echo "=== 4. Pulizia disco: rimozione dei 218 GB grezzi ==="
rm -f trj_tel26_raw.xtc

echo "=== COMPLETATO CON SUCCESSO! ==="
