import sys
import math

def dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

atoms = {}
with open("tutorials/tel22/video_tel22.pdb", "r") as f:
    frame = 0
    for line in f:
        if line.startswith("MODEL"):
            frame += 1
            if frame > 1: break
            atoms[frame] = {}
        if line.startswith("ATOM") and "DG" in line:
            name = line[12:16].strip()
            # resi is 1..220 for 10 molecules
            resi = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if resi not in atoms[frame]:
                atoms[frame][resi] = []
            if name.startswith("B"):
                atoms[frame][resi].append((x,y,z))

frame = 1
# molecule 1 is resi 1..22, mol 2 is 23..44, etc.
for mol in range(10):
    offset = mol * 22
    g_centers = {}
    for r in [2,3,4,8,9,10,14,15,16,20,21,22]:
        resi = offset + r
        if resi in atoms[frame] and len(atoms[frame][resi]) > 0:
            coords = atoms[frame][resi]
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
            cz = sum(c[2] for c in coords) / len(coords)
            g_centers[r] = (cx, cy, cz)
    
    if len(g_centers) < 12: continue
    
    print(f"Molecule {mol+1}:")
    tracts = [[2,3,4], [8,9,10], [14,15,16], [20,21,22]]
    for t in tracts:
        for r in t:
            closest = []
            for other_t in tracts:
                if other_t != t:
                    best_d = 999
                    best_r = -1
                    for other_r in other_t:
                        d = dist(g_centers[r], g_centers[other_r])
                        if d < best_d:
                            best_d = d
                            best_r = other_r
                    closest.append((best_d, best_r))
            closest.sort()
            print(f"  Res {r} closest in other tracts: {[other_r for d, other_r in closest]}")
