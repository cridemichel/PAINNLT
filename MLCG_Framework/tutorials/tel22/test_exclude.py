import json

with open("tel22_topology.json", "r") as f:
    config = json.load(f)

exclusions = set()
for b in config.get("bonds", []):
    mol_i = b["mol_i"]
    mol_j = b["mol_j"]
    exclusions.add((min(mol_i, mol_j), max(mol_i, mol_j)))

print(f"Number of bonded exclusions: {len(exclusions)}")
