import json

# Carica topology
with open("tel22_topology.json", "r") as f:
    topo = json.load(f)

# Carica priors IBI attuali
with open("cg_priors.json", "r") as f:
    priors = json.load(f)

# Sostituisci bonds, angles, dihedrals nella topology con quelli dei priors (che sono "tabulated" e puntano ai file IBI)
topo["bonds"] = priors.get("bonds", [])
topo["angles"] = priors.get("angles", [])
topo["dihedrals"] = priors.get("dihedrals", [])

# Salva
with open("tel22_topology.json", "w") as f:
    json.dump(topo, f, indent=4)

print("Topology updated with IBI bonds/angles/dihedrals!")
