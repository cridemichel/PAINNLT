#!/bin/bash

echo "[1/3] Compilazione dello script di estrazione C++..."
cd ../../training/build
cmake ..
make eval_parity -j4

cd ../../tutorials/tel22_IBI

echo ""
echo "[2/3] Estrazione delle Forze dal Validation Set..."
# Lanciamo l'estrattore usando il dataset e il modello appena allenati
../../training/build/eval_parity --dataset tel22_dataset_ibi.bin --model tel22_model_ibi.pt --config tel22_training_config.json

echo ""
echo "[3/3] Generazione del Grafico (Parity Plot) e metriche..."
# L'estrattore ha generato parity_forces.csv, ora Python lo plotta
../../mlcg_venv/bin/python3 plot_parity.py

echo ""
echo "Finito! Controlla l'immagine 'parity_plot.png'."
