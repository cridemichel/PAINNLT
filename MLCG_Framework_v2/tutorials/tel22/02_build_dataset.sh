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
    and meta.get("pair_source") == "explicit_topology_pairs_v2"
    and isinstance(meta.get("direct_pairs"), list)
    and isinstance(meta.get("one_three_pairs"), list)
):
    raise SystemExit(
        "[ERRORE] cg_priors.json non dichiara liste esplicite WCA 1-2/1-3 v2."
    )

def bond_excludes_wca(bond):
    return bool(bond.get("exclude_wca", str(bond.get("type", "harmonic")).lower() != "morse"))

direct = set()
for bond in config.get("bonds", []):
    if not bond_excludes_wca(bond):
        continue
    i, j = int(bond["mol_i"]), int(bond["mol_j"])
    if i != j:
        direct.add((min(i, j), max(i, j)))

one_three = set()
for angle in config.get("angles", []):
    if not bool(angle.get("exclude_wca", True)):
        continue
    i, k = int(angle["mol_i"]), int(angle["mol_k"])
    if i != k:
        key = (min(i, k), max(i, k))
        if key not in direct:
            one_three.add(key)

expected_direct = len(direct)
expected_one_three = len(one_three)
actual_direct = int(meta.get("direct_pair_count", -1))
actual_one_three = int(meta.get("one_three_pair_count", -1))

stored_direct = {tuple(sorted(map(int, pair))) for pair in meta["direct_pairs"]}
stored_one_three = {tuple(sorted(map(int, pair))) for pair in meta["one_three_pairs"]}
if (actual_direct, actual_one_three) != (expected_direct, expected_one_three):
    raise SystemExit(
        "[ERRORE] Conteggi WCA exclusions incoerenti: "
        f"cg_priors=({actual_direct}, {actual_one_three}), "
        f"topologia=({expected_direct}, {expected_one_three})."
    )
if stored_direct != direct or stored_one_three != one_three:
    raise SystemExit("[ERRORE] Le liste esplicite WCA non coincidono con la topologia sorgente.")

morse_pairs = {
    tuple(sorted((int(b["mol_i"]), int(b["mol_j"]))))
    for b in config.get("bonds", [])
    if str(b.get("type", "")).lower() == "morse"
}
if stored_direct & morse_pairs:
    raise SystemExit("[ERRORE] Un restraint Morse TEL22 sta ancora disabilitando il WCA.")

angle_default_site = config.get("prior_geometry", {}).get("default_angle_site", None)
if angle_default_site != 0:
    raise SystemExit("[ERRORE] TEL22 richiede prior_geometry.default_angle_site=0.")
for idx, angle in enumerate(priors.get("angles", [])):
    if (angle.get("site_i"), angle.get("site_j"), angle.get("site_k")) != (0, 0, 0):
        raise SystemExit(f"[ERRORE] Angle prior {idx} non e site0-based: {angle}")

print(
    "[INFO] WCA exclusion metadata verificata: "
    f"{actual_direct} coppie topologiche 1-2, {actual_one_three} coppie 1-3; "
    "Morse restraints mantengono il WCA."
)
print("[INFO] Backbone angle priors verificati sulla catena site0-site0-site0.")
PY

echo "Dataset tel22_dataset.bin generato con successo."
