import torch
import numpy as np

dataset = torch.load("cg_dataset.bin")
types = set()
for d in dataset:
    types.update(d['z'].numpy().tolist())
print(f"Types in dataset: {types}")
