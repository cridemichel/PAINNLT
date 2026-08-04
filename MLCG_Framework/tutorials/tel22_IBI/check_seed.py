import json

with open("cg_priors_seed.json", "r") as f:
    data = json.load(f)

for k in ["bonds", "angles", "dihedrals"]:
    print(f"{k}: {[b.get('type') for b in data.get(k, [])[:3]]}")
