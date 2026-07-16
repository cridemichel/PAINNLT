import numpy as np
import sys
import os

with open("md_whole.trr", "rb") as f:
    pass

import MDAnalysis as mda
u = mda.Universe("md.gro", "md_whole.trr")
print("Has forces?", u.trajectory[0].has_forces)
