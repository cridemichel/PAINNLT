#!/bin/bash
set -e
echo "[3/4] Addestramento della Rete Neurale (PaiNN) sul Dataset Residuo"
../../training/build/train_painn tel22_residual_dataset.bin tel22_model_ibi.pt tel22_training_config.json
