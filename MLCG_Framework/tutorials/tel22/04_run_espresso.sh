#!/bin/bash
set -e

echo "======================================================"
echo " 04. ESPRESSO MD SIMULATION "
echo "======================================================"

if [ ! -f "tel22_model.pt" ]; then
    echo "Errore: Modello tel22_model.pt non trovato! Hai eseguito 03_train_model.sh?"
    exit 1
fi

echo "Avvio la simulazione ESPResSo usando il pypresso di sistema."
echo "Per personalizzare, apri e modifica lo script run_cg_md.py."

# Assumiamo che l'utente l'abbia compilato in /path/to/espresso/build/pypresso
# In un caso reale, qui ci andrebbe il comando pypresso. Per ora, lascio a lui.
echo "Esegui questo comando con il tuo eseguibile pypresso:"
echo "/path/to/tuo/espresso/build/pypresso ../../simulation/run_cg_md.py \\"
echo "    --model tel22_model.pt \\"
echo "    --config tel22_training_config.json \\"
echo "    --priors cg_priors.json \\"
echo "    --rb_info rigid_bodies_info.json \\"
echo "    --dataset tel22_dataset.bin \\"
echo "    --steps 10000 --dt 0.002"
