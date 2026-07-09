import json

with open("tel22_topology.json", "r") as f:
    data = json.load(f)

data["wca_sigma"] = 0.3
data["wca_epsilon"] = 1.0

with open("tel22_topology.json", "w") as f:
    json.dump(data, f, indent=4)
