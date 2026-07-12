#!/bin/bash
set -e

echo "=== TUTORIAL MLCG FRAMEWORK: TEL22 ==="
echo "Questo script lancia l'intera pipeline di Machine Learning Coarse-Graining in automatico."

./01_run_gromacs.sh
./02_build_dataset.sh
./03_train_model.sh
./04_equilibrate.sh
./05_run_espresso.sh

echo "Pipeline Completata con Successo!"
