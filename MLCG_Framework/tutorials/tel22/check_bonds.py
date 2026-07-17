import json

with open("tel22_topology.json", "r") as f:
    config = json.load(f)

for b in config.get("bonds", [])[:10]:
    print(b)
