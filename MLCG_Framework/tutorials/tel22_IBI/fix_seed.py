import json

with open("cg_priors.json", "r") as f:
    data = json.load(f)

for k in ["bonds", "angles", "dihedrals"]:
    for item in data.get(k, []):
        if item.get("type") == "tabulated":
            item["type"] = "ibi"

with open("cg_priors_seed.json", "w") as f:
    json.dump(data, f, indent=4)
