#!/bin/bash
#SBATCH -N1
#SBATCH --job-name=postprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -t 02:00:00
#SBATCH -A IscrB_G4MES

module load profile/chem-phys
module load fftw/3.3.10--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load gromacs/2023.3--gcc--12.2.0-cuda-12.1

export OMP_NUM_THREADS=8

echo "=== 1. Unione delle 3 parti di traiettoria ==="
gmx trjcat -f prod-1.part0001.xtc prod-2.part0002.xtc prod-3.part0003.xtc -o trj_tel26_raw.xtc -cat

echo "=== 2. Estrazione struttura di riferimento DNA ==="
echo "1" | gmx trjconv -s md_hybrid_tel26.tpr -f equil_hybrid_tel26.gro -o tel26_dna_ref.gro

echo "=== 3. Ricentratura PBC ed estrazione del solo DNA ==="
echo -e "1\n1" | gmx trjconv -s md_hybrid_tel26.tpr -f trj_tel26_raw.xtc -o trj_tel26_dna_pbc.xtc -pbc mol -center

echo "=== 4. Fit rototraslazionale per VMD e analisi ==="
echo -e "1\n1" | gmx trjconv -s md_hybrid_tel26.tpr -f trj_tel26_dna_pbc.xtc -o trj_tel26_dna_fit.xtc -fit rot+trans

echo "=== 5. Calcolo RMSD rispetto al frame iniziale ==="
echo -e "1\n1" | gmx rms -s md_hybrid_tel26.tpr -f trj_tel26_dna_pbc.xtc -o rmsd_dna.xvg -tu ns

echo "=== POST-PROCESSING COMPLETATO CON SUCCESSO! ==="
