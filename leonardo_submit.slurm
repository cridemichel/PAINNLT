#!/bin/bash
#SBATCH --job-name=painn_espresso
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --account=<TUO_ACCOUNT_CINECA>
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# 1. Carica il modulo di Apptainer
module load apptainer

# 2. Definisci il path dell'immagine (sif) e della tua cartella di lavoro
IMAGE_PATH="/leonardo/home/userexternal/<username>/painn_env.sif"
WORK_DIR="/leonardo/home/userexternal/<username>/PAINNLT"

# 3. Spostati nella cartella di lavoro
cd $WORK_DIR

# 4. Esegui il comando desiderato tramite Apptainer
# Spiegazione dei flag:
# --nv : Abilita il supporto NVIDIA GPU nel container (fondamentale per CUDA/PyTorch)
# --bind : Monta la cartella di lavoro all'interno del container in /app
# 
# ESEMPIO 1: Compilare ESPResSo e il training tool (decommentare se serve compilare)
# srun apptainer exec --nv --bind $WORK_DIR:/app $IMAGE_PATH bash -c "cd /app/espresso/build && cmake .. -DCMAKE_PREFIX_PATH=/opt/conda/lib/python3.10/site-packages/torch/share/cmake/Torch && make -j8"

# ESEMPIO 2: Generare il dataset (conversione traiettoria Gromacs)
# srun apptainer exec --nv --bind $WORK_DIR:/app $IMAGE_PATH bash -c "cd /app/python_scripts && python convert_gro2bin.py --topology ../sim.tpr --trajectory ../sim.trr --output ../dataset.bin"

# ESEMPIO 3: Lanciare il Training di PaiNN
srun apptainer exec --nv --bind $WORK_DIR:/app $IMAGE_PATH bash -c "cd /app/build && ./cg_painn_train"

# ESEMPIO 4: Lanciare la simulazione ESPResSo
# srun apptainer exec --nv --bind $WORK_DIR:/app $IMAGE_PATH bash -c "cd /app && ./espresso/build/pypresso espresso_integration/simulate_tel22.py"
