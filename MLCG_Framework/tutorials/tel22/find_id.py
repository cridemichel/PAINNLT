import json
with open("tel22_topology.json", "r") as f:
    topo = json.load(f)

vs_count = 36 # 0 to 35 are COMs
for i, mol in enumerate(topo["mapping"]):
    num_vs = len(mol["virtual_sites"])
    for j in range(num_vs):
        if vs_count == 835 or vs_count == 842:
            print(f"ID {vs_count} is mol {i}, site {j}")
        vs_count += 1
