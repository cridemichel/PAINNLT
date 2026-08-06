#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CORE_DIR="$FRAMEWORK_ROOT/espresso/src/core/nonbonded_interactions"
PYTHON_DIR="$FRAMEWORK_ROOT/espresso/src/python/espressomd"

if [[ ! -d "$CORE_DIR" || ! -d "$PYTHON_DIR" ]]; then
    echo "[ERROR] ESPResSo source tree not found under: $FRAMEWORK_ROOT/espresso" >&2
    exit 1
fi

# PaiNN_Architecture.hpp may be a symlink in the plugin directory. Copy the
# authoritative training header directly so regular-file and symlink layouts
# produce the same ESPResSo sources.
cp -f "$FRAMEWORK_ROOT/training/PaiNN_Architecture.hpp" "$CORE_DIR/"
cp -f "$SCRIPT_DIR/PaiNN_ML_Potential.hpp" "$CORE_DIR/"
cp -f "$SCRIPT_DIR/PaiNN_ML_Potential.cpp" "$CORE_DIR/"
cp -f "$SCRIPT_DIR/painn.pyx" "$PYTHON_DIR/"

echo "[PASS] PaiNN plugin sources synchronized with ESPResSo."
