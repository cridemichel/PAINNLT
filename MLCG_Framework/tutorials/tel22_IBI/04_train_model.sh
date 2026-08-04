#!/bin/bash
set -e
echo "[4/5] Addestramento della Rete Neurale (PaiNN) sul Dataset Residuo"
../../training/build/train_painn tel22_dataset_ibi_v2.bin tel22_model_ibi_v2.pt tel22_training_config.json
