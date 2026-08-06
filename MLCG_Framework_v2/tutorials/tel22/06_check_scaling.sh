#!/bin/bash
set -e

echo "======================================================"
echo " 06. ENERGY CONSERVATION SCALING CHECK "
echo "======================================================"

if [ ! -f "equilibrated.npz" ] || [ ! -f "tel22_model.pt" ]; then
    echo "Errore: equilibrated.npz o tel22_model.pt non trovati. Esegui gli step precedenti."
    exit 1
fi

echo "Avvio i test NVE a dt incrementali..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
/Users/demichel/PYTHON/bin/python check_scaling.py 

echo "[SUCCESS] Scaling check completato. Controlla il grafico generato."
