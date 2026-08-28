import sys
import math

def dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

atoms = {}
# B1 is the center of the base roughly. Or we can just average all 5 base beads (B1..B5) for each Guanine.
with open("tutorials/tel22/video_tel22.pdb", "r") as f:
    frame = 0
    for line in f:
        if line.startswith("MODEL"):
            frame += 1
            atoms[frame] = {}
        if line.startswith("ATOM") and "DG A" in line:
            name = line[12:16].strip()
            resi = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if resi not in atoms[frame]:
                atoms[frame][resi] = []
            if name.startswith('B'):
                atoms[frame][resi].append((x,y,z))

# We only care about the last frame, Molecule A (resi 1..22)
frame = max(atoms.keys())
g_centers = {}
for resi, coords in atoms[frame].items():
    if len(coords) > 0:
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        g_centers[resi] = (cx, cy, cz)

# Find distances between G bases
resis = list(g_centers.keys())
resis.sort()

# Build an adjacency graph based on distance < 1.0 nm (10 Angstroms)
# Hoogsteen base pairs in CG model should be fairly close.
edges = []
for i in range(len(resis)):
    for j in range(i+1, len(resis)):
        r1 = resis[i]
        r2 = resis[j]
        d = dist(g_centers[r1], g_centers[r2])
        edges.append((d, r1, r2))

edges.sort()
print("Closest pairs:")
for e in edges[:15]:
    print(f"{e[1]} - {e[2]}: {e[0]:.2f}")

