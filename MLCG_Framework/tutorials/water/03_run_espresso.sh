#!/bin/bash
set -e

echo "======================================================"
echo " 03. ESPRESSO MD SIMULATION (WATER) "
echo "======================================================"

if [ ! -f "my_water_model.pt" ]; then
    echo "Errore: Modello my_water_model.pt non trovato! Hai eseguito 02_train_model.sh?"
    exit 1
fi

echo "Avvio la simulazione ESPResSo usando il pypresso di sistema."
echo "Esegui questo comando con il tuo eseguibile pypresso per far partire la validazione:"
echo "/path/to/tuo/espresso/build/pypresso verify_water.py"
