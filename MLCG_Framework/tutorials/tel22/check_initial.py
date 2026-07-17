import sys
sys.path.append("../../preprocessing")
from dataset import CGDataset
import json
import numpy as np

ds = CGDataset("tel22_dataset.bin")
frame0 = ds.read_frame(0)

with open('cg_bonds.json', 'r') as f:
    bonds = json.load(f)

for idx, b in enumerate(bonds):
    if b['type'] == 'tabulated' and b['name'] == 'bb_G_G':
        d = b['params']
        mol_i = d['mol_i']
        mol_j = d['mol_j']
        site_i = d.get('site_i', -1)
        site_j = d.get('site_j', -1)
        
        pos_i = frame0.vs_positions[mol_i][site_i] if site_i != -1 else frame0.com_positions[mol_i]
        pos_j = frame0.vs_positions[mol_j][site_j] if site_j != -1 else frame0.com_positions[mol_j]
        
        dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
        print(f"bb_G_G {mol_i}:{site_i} - {mol_j}:{site_j} initial dist = {dist:.4f} nm")

