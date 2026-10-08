#!/bin/bash
#SBATCH -N1
#SBATCH --job-name Tel26-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -t 24:00:00
#SBATCH -A IscrB_G4MES 

module load profile/chem-phys
module load fftw/3.3.10--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0
module load gromacs/2023.3--gcc--12.2.0-cuda-12.1


export OMP_NUM_THREADS=32
export GMX_GPU_DD_COMMS=true
export GMX_GPU_PME_PP_COMMS=true
export GMX_FORCE_UPDATE_DEFAULT_GPU=true


gmx mdrun -s md_hybrid_tel26.tpr -deffnm prod-1 -cpi prod-1.cpt -noappend -ntmpi 1 -ntomp 32 -v -maxh 24.0  -nb gpu -pme gpu 
