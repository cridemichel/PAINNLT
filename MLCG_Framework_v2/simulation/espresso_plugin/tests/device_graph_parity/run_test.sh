#!/usr/bin/env bash
# Test di parita' legacy/device del plugin PaiNN senza ESPResSo (header finti
# in stubs/, modello PaiNN con pesi casuali).  Serve solo libtorch: usa quella
# del torch di Python.  Uso:  bash run_test.sh   (stampa |dF|, dE e un FD)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN="$(cd "$HERE/../.." && pwd)"
FRAMEWORK="$(cd "$PLUGIN/../.." && pwd)"
TP="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
WORK="$(mktemp -d)"
cp "$PLUGIN/PaiNN_ML_Potential.cpp" "$PLUGIN/PaiNN_ML_Potential.hpp" "$WORK/"
cp "$FRAMEWORK/training/PaiNN_Architecture.hpp" "$WORK/"
g++ -std=c++20 -O1 -I "$HERE/stubs" -I "$WORK" -I "$TP/include" \
    -I "$TP/include/torch/csrc/api/include" \
    "$HERE/test_parity.cpp" "$WORK/PaiNN_ML_Potential.cpp" \
    -L "$TP/lib" -Wl,-rpath,"$TP/lib" -ltorch -ltorch_cpu -lc10 -o "$WORK/test_parity"
for precision in float64 float32; do
    "$WORK/test_parity" "$precision" 2>&1 | grep -E "precision=|max\||net force|marker|FD axis"
done
