#!/bin/bash
set -e
echo "======================================"
echo "    PIPELINE TEL22 IBI + ML"
echo "======================================"

./01_build_dataset.sh
./02_run_ibi.sh
./03_subtract_ibi.sh
./04_train_model.sh
./05_equilibrate.sh
./06_run_espresso.sh

echo "======================================"
echo "     TUTTI I TEST COMPLETATI!"
echo "======================================"
