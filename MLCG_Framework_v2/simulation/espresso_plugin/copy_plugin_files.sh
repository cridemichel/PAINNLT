#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ESPRESSO_ROOT="${ESPRESSO_SRC:-$FRAMEWORK_ROOT/espresso}"
CORE_DIR="$ESPRESSO_ROOT/src/core/nonbonded_interactions"
PYTHON_DIR="$ESPRESSO_ROOT/src/python/espressomd"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$CORE_DIR" || ! -d "$PYTHON_DIR" ]]; then
    echo "[ERROR] ESPResSo source tree not found under: $ESPRESSO_ROOT" >&2
    echo "        Set ESPRESSO_SRC=/path/to/espresso if it lives elsewhere." >&2
    exit 1
fi
copy_if_different() {
    local src="$1"
    local dst="$2"

    if [[ -e "$dst" ]] && cmp -s "$src" "$dst"; then
        printf '[SKIP] Already identical: %s\n' "$dst"
        return 0
    fi

    cp -f "$src" "$dst"
}

copy_if_different "$FRAMEWORK_ROOT/training/PaiNN_Architecture.hpp" \
    "$CORE_DIR/PaiNN_Architecture.hpp"

copy_if_different "$SCRIPT_DIR/PaiNN_ML_Potential.hpp" \
    "$CORE_DIR/PaiNN_ML_Potential.hpp"

copy_if_different "$SCRIPT_DIR/PaiNN_ML_Potential.cpp" \
    "$CORE_DIR/PaiNN_ML_Potential.cpp"

copy_if_different "$SCRIPT_DIR/painn.pyx" \
    "$PYTHON_DIR/painn.pyx"
#
# Install the conservative pairwise Morse bond in the core, ScriptInterface,
# and Python interface. The installer is idempotent and fails closed if the
# ESPResSo source layout is not recognized.
"$PYTHON_BIN" "$SCRIPT_DIR/install_analytic_morse_bond.py" \
    --espresso-root "$ESPRESSO_ROOT"

# Extend ESPResSo's stock non-bonded Morse with an optional smooth switching
# tail.  Stock behavior is preserved when switch_start is left at -1.
"$PYTHON_BIN" "$SCRIPT_DIR/install_switched_morse_nonbonded.py" \
    --espresso-root "$ESPRESSO_ROOT"

printf '[PASS] PaiNN plugin, analytic MorseBond diagnostic, and switched non-bonded Morse synchronized with ESPResSo: %s\n' "$ESPRESSO_ROOT"
printf '[INFO] Reconfigure before rebuilding so changes to build/myconfig.hpp are picked up: cmake -S %s -B %s/build\n' "$ESPRESSO_ROOT" "$ESPRESSO_ROOT"
