import json

with open("tel22_topology.json") as f:
    data = json.load(f)

# Find all Morse bonds
morse_bonds = []
for b in data.get("bonds", []):
    if isinstance(b, dict) and b.get("type") == "morse":
        morse_bonds.append((b["mol_i"], b["mol_j"]))

print(f"Total morse bonds: {len(morse_bonds)}")

# Group them into connected components (cliques)
from collections import defaultdict
adj = defaultdict(set)
for u, v in morse_bonds:
    adj[u].add(v)
    adj[v].add(u)

visited = set()
cliques = []
for node in adj:
    if node not in visited:
        # Simple BFS
        q = [node]
        comp = set()
        while q:
            curr = q.pop(0)
            if curr not in comp:
                comp.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in comp:
                        q.append(neighbor)
        visited.update(comp)
        cliques.append(sorted(list(comp)))

for i, c in enumerate(cliques):
    print(f"Tetrad {i+1}: {c}")

