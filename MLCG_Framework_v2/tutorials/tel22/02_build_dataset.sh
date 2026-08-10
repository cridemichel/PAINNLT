#!/bin/bash
set -euo pipefail

echo "======================================================"
echo " 02. PREPROCESSING AND DATASET GENERATION "
echo "======================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILDER="${FRAMEWORK_ROOT}/preprocessing/build_cg_dataset.py"
PYTHON_BIN="${PYTHON_BIN:-/Users/demichel/PYTHON/bin/python}"

cd "${SCRIPT_DIR}"

if [ ! -f "md_whole.trr" ] || [ ! -f "md.gro" ]; then
    echo "Errore: md_whole.trr o md.gro non trovati! Hai eseguito 01_run_gromacs.sh con trjconv -force?"
    exit 1
fi

if [ ! -f "tel22_topology.json" ]; then
    echo "Errore: tel22_topology.json non trovato in ${SCRIPT_DIR}."
    exit 1
fi

if [ ! -f "${BUILDER}" ]; then
    echo "Errore: preprocessing builder non trovato: ${BUILDER}"
    exit 1
fi

# Fail fast if an old/partial preprocessing source is selected accidentally.
if ! grep -q "def build_wca_topology_exclusions" "${BUILDER}" || \
   ! grep -q "nonbonded candidate pairs" "${BUILDER}"; then
    echo "Errore: ${BUILDER} non contiene la policy WCA 1-2/1-3 completa."
    echo "Non genero il dataset con un preprocessing legacy o parzialmente patchato."
    exit 1
fi

echo "[INFO] Dataset builder: ${BUILDER}"
echo "[INFO] Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" "${BUILDER}" \
    --topology md.gro \
    --trajectory md_whole.trr \
    --config tel22_topology.json \
    --output tel22_dataset.bin

# Verify that the generated priors encode the same TEL22 topology exclusions
# expected from the source topology. This catches stale/legacy builders even if
# they happen to complete without raising an exception.
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("tel22_topology.json").read_text())
priors = json.loads(Path("cg_priors.json").read_text())
rb_info = json.loads(Path("rigid_bodies_info.json").read_text())
meta = priors.get("wca_exclusions", {})

if float(config.get("decoy_target_fraction", 0.0)) != 0.0:
    raise SystemExit(
        "[ERRORE] I legacy decoy whole-frame senza loss mask devono essere disabilitati "
        "(decoy_target_fraction=0)."
    )

for resname, info in rb_info.items():
    sites = info.get("sites", {})
    if len(sites) == 1:
        site_name, site = next(iter(sites.items()))
        rel = [float(v) for v in site.get("relative_pos_nm", [])]
        if len(rel) != 3 or sum(v*v for v in rel) > 1.0e-12:
            raise SystemExit(
                f"[ERRORE] Corpo one-site {resname}/{site_name} non centrato sul COM: {rel}."
            )

if not (
    meta.get("exclude_12") is True
    and meta.get("exclude_13") is True
    and meta.get("scope") == "molecule_pair_all_sites"
):
    raise SystemExit(
        "[ERRORE] cg_priors.json non dichiara la policy WCA 1-2/1-3 richiesta."
    )

direct = set()
for bond in config.get("bonds", []):
    i, j = int(bond["mol_i"]), int(bond["mol_j"])
    if i != j:
        direct.add((min(i, j), max(i, j)))

one_three = set()
for angle in config.get("angles", []):
    i, k = int(angle["mol_i"]), int(angle["mol_k"])
    if i != k:
        key = (min(i, k), max(i, k))
        if key not in direct:
            one_three.add(key)

expected_direct = len(direct)
expected_one_three = len(one_three)
actual_direct = int(meta.get("direct_pair_count", -1))
actual_one_three = int(meta.get("one_three_pair_count", -1))

if (actual_direct, actual_one_three) != (expected_direct, expected_one_three):
    raise SystemExit(
        "[ERRORE] Conteggi WCA exclusions incoerenti: "
        f"cg_priors=({actual_direct}, {actual_one_three}), "
        f"topologia=({expected_direct}, {expected_one_three})."
    )

print(
    "[INFO] WCA exclusion metadata verificata: "
    f"{actual_direct} coppie 1-2, {actual_one_three} coppie 1-3."
)
PY

echo "Dataset tel22_dataset.bin generato con successo."
