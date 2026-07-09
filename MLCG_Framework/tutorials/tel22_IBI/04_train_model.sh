#!/bin/bash
set -e
echo "[4/5] Addestramento della Rete Neurale (PaiNN) sul Dataset Residuo"
../../training/build/train_painn tel22_dataset_ibi.bin tel22_model_ibi.pt tel22_training_config.json
