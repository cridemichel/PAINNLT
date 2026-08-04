import json

with open("cg_priors.json", "r") as f:
    data = json.load(f)

for a in data.get("angles", [])[:5]:
    print(a.get("name"))
