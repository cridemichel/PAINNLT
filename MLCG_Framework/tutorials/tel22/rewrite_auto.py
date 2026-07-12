import json

SEQ = list("AGGGTTAGGGTTAGGGTTAGGG")

with open("tel22_topology.json", "r") as f:
    data = json.load(f)

new_bonds = []
for b in data.get("bonds", []):
    if isinstance(b, list):
        mol_i, mol_j, site_i, site_j = b
        res_i = SEQ[mol_i % 22]
        res_j = SEQ[mol_j % 22]
        new_bonds.append({
            "mol_i": mol_i,
            "mol_j": mol_j,
            "site_i": site_i,
            "site_j": site_j,
            "type": "harmonic",
            "k": "auto",
            "r0": "auto",
            "name": f"bb_{res_i}_{res_j}"
        })
    elif isinstance(b, dict):
        new_bonds.append(b)

data["bonds"] = new_bonds

new_angles = []
for a in data.get("angles", []):
    mol_i, mol_j, mol_k = a["mol_i"], a["mol_j"], a["mol_k"]
    res_i = SEQ[mol_i % 22]
    res_j = SEQ[mol_j % 22]
    res_k = SEQ[mol_k % 22]
    new_a = {
        "mol_i": mol_i,
        "mol_j": mol_j,
        "mol_k": mol_k,
        "type": "harmonic",
        "k": "auto",
        "theta0": "auto",
        "name": f"ang_{res_i}_{res_j}_{res_k}"
    }
    if "site_i" in a: new_a["site_i"] = a["site_i"]
    if "site_j" in a: new_a["site_j"] = a["site_j"]
    if "site_k" in a: new_a["site_k"] = a["site_k"]
    new_angles.append(new_a)

data["angles"] = new_angles

new_dihedrals = []
for d in data.get("dihedrals", []):
    mol_i, mol_j, mol_k, mol_l = d["mol_i"], d["mol_j"], d["mol_k"], d["mol_l"]
    res_i = SEQ[mol_i % 22]
    res_j = SEQ[mol_j % 22]
    res_k = SEQ[mol_k % 22]
    res_l = SEQ[mol_l % 22]
    new_d = {
        "mol_i": mol_i,
        "mol_j": mol_j,
        "mol_k": mol_k,
        "mol_l": mol_l,
        "type": "cosine",
        "k": "auto",
        "phi0": "auto",
        "n": 1,
        "name": f"dih_{res_i}_{res_j}_{res_k}_{res_l}"
    }
    if "site_i" in d: new_d["site_i"] = d["site_i"]
    if "site_j" in d: new_d["site_j"] = d["site_j"]
    if "site_k" in d: new_d["site_k"] = d["site_k"]
    if "site_l" in d: new_d["site_l"] = d["site_l"]
    new_dihedrals.append(new_d)

data["dihedrals"] = new_dihedrals

with open("tel22_topology.json", "w") as f:
    json.dump(data, f, indent=4)

print("Riscrittura AUTO completata con successo! Sovrascritto tel22_topology.json")
