import numpy as np
import argparse
import sys
import os

def check_equilibrated(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        sys.exit(1)
        
    data = np.load(file_path)
    print(f"Keys: {data.files}")
    print(f"Pos shape: {data['pos'].shape}")
    print(f"Pos nan?: {np.isnan(data['pos']).any()}")
    print(f"Pos min/max: {np.min(data['pos'])}, {np.max(data['pos'])}")

if __name__ == "__main__":
    check_equilibrated("equilibrated.npz")
