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
# ESPResSo elenca i sorgenti del core esplicitamente in target_sources(): un
# file copiato nella directory non viene compilato se non compare in quella
# lista.  Senza questo passo il core non contiene global_painn_potential e
# l'import del modulo muore con "undefined symbol" -- su Linux, dove il link
# e' stretto; su macOS il sintomo slitta al runtime per via di
# -undefined dynamic_lookup.
"$PYTHON_BIN" "$SCRIPT_DIR/install_painn_core_sources.py" \
    --espresso-root "$ESPRESSO_ROOT"

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

# myconfig.hpp: ESPResSo decide a COMPILAZIONE quali interazioni esistono, e il
# suo default non include MORSE -- su cui la pipeline CG fonda i contatti fra
# guanine di una tetrade.  Senza, tutto il codice Morse (estensione switched
# compresa, che sta dentro #ifdef MORSE) viene compilato via e la produzione
# muore molto piu' tardi, dopo dataset e training.
#
# Va nella radice dei sorgenti e non nella build directory: bootstrap_leonardo
# rigenera la build quando cambia il compilatore, e li' il file sparirebbe.
copy_if_different "$SCRIPT_DIR/myconfig.hpp" "$ESPRESSO_ROOT/myconfig.hpp"
if [[ -f "$ESPRESSO_ROOT/build/myconfig.hpp" ]] && \
   ! cmp -s "$SCRIPT_DIR/myconfig.hpp" "$ESPRESSO_ROOT/build/myconfig.hpp"; then
    # Una copia nella build directory ha la precedenza su quella dei sorgenti:
    # se e' rimasta da prima e differisce, vince lei e il file appena
    # installato non ha alcun effetto.
    echo "[WARN] $ESPRESSO_ROOT/build/myconfig.hpp differisce e ha la precedenza:" >&2
    echo "       lo allineo a quello del framework." >&2
    cp -f "$SCRIPT_DIR/myconfig.hpp" "$ESPRESSO_ROOT/build/myconfig.hpp"
fi

printf '[PASS] PaiNN plugin, analytic MorseBond diagnostic, switched non-bonded Morse e myconfig synchronized with ESPResSo: %s\n' "$ESPRESSO_ROOT"
printf '[INFO] Reconfigure before rebuilding so changes to build/myconfig.hpp are picked up: cmake -S %s -B %s/build\n' "$ESPRESSO_ROOT" "$ESPRESSO_ROOT"
