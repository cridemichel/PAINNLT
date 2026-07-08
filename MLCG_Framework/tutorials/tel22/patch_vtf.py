import re

with open("cg_trajectory.vtf", "r") as f:
    lines = f.readlines()

# Find the last bond line
last_bond_idx = -1
for i, line in enumerate(lines):
    if line.startswith("timestep"):
        last_bond_idx = i
        break

if last_bond_idx == -1:
    print("Could not find timestep")
    exit(1)

# We need to know the COM and VS.
# In ESPResSo, the particles were added sequentially.
# From the VTF header:
# atom 0 (type 5) -> DUMMY_COM
# atom 1 (type 0) -> VS
# atom 2 (type 5) -> DUMMY_COM
# atom 3 (type 2) -> VS
# atom 4 (type 3) -> VS
# atom 5 (type 4) -> VS
# atom 6 (type 5) -> DUMMY_COM ... wait, type 5 is DUMMY COM?
