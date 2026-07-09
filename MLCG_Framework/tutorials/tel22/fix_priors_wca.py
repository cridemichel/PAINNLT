import json

with open("cg_priors.json", "r") as f:
    data = json.load(f)

if "wca" in data:
    data["wca"]["sigma"] = 0.6

with open("cg_priors.json", "w") as f:
    json.dump(data, f, indent=4)
