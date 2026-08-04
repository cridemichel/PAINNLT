#!/bin/bash
set -euo pipefail

echo "[1/3] Compilazione dello script di estrazione C++..."
cd ../../training/build
cmake ..
cmake --build . --target eval_parity -j4

cd ../../tutorials/tel22_IBI

echo ""
echo "[2/3] Estrazione delle forze dal validation set usato nel training..."
../../training/build/eval_parity     --dataset tel22_dataset_ibi_v2.bin     --model tel22_model_ibi_v2.pt     --config tel22_training_config.json

echo ""
echo "[3/3] Generazione del parity plot e delle metriche..."
uv run python plot_parity.py

echo ""
echo "Finito! Controlla l'immagine 'parity_plot.png'."
