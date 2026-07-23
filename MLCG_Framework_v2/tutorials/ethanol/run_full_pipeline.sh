#!/bin/bash
set -e

echo "=== TUTORIAL MLCG FRAMEWORK: ETHANOL ==="
echo "Questo script lancia l'intera pipeline in automatico."

./01_build_dataset.sh
./02_train_model.sh
./03_run_espresso.sh

echo "Pipeline Completata con Successo!"
