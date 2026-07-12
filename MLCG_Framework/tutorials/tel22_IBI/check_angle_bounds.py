import numpy as np
import json
import glob

# Check the bounds where force is NOT large (inside the valid region)
for file in glob.glob("ibi_priors/angle_tabulated_*.dat"):
    data = np.loadtxt(file)
    x = data[:, 0]
    f = data[:, 2]
    # find where |f| < 1000
    valid_idx = np.where(np.abs(f) < 1000)[0]
    if len(valid_idx) > 0:
        first, last = valid_idx[0], valid_idx[-1]
        print(f"{file}: Valid region [{x[first]:.2f}, {x[last]:.2f}]")
    else:
        print(f"{file}: NO VALID REGION!?")

