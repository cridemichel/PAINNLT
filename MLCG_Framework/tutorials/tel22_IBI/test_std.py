import sys
import os
import torch

sys.path.insert(0, os.path.abspath('../../training/src'))
import dataset

ds = dataset.CGDataset('../tel22/tel22_dataset.bin')
print('Original Force std:', ds[0].force.std().item())
print('Original Force std (all):', torch.cat([d.force for d in ds]).std().item())
