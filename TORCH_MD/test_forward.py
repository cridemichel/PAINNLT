import torch
import numpy as np
from torchmdnet.models.model import create_model

# mock arguments
args = {
    'model': 'equivariant-transformer',
    'hidden_channels': 128,
    'embedding_dimension': 128,
    'num_layers': 3,
    'num_rbf': 50,
    'rbf_type': 'expnorm',
    'trainable_rbf': True,
    'activation': 'silu',
    'attn_activation': 'silu',
    'num_heads': 8,
    'distance_influence': 'both',
    'neighbor_embedding': True,
    'cutoff_lower': 0.0,
    'cutoff_upper': 0.9,
    'max_z': 100,
    'max_num_neighbors': 32,
    'aggr': 'add',
    'num_rbf': 50,
    'trainable_rbf': True,
    'rbf_type': 'expnorm',
    'distance_influence': 'both',
    'reduce_op': 'add',
    'derivative': True,
    'prior_model': None,
    'atom_filter': -1,
    'check_errors': True,
    'precision': 32
}

model = create_model(args)
pos = torch.tensor(np.load("dataset_pos.npy")[0])
z = torch.tensor(np.load("dataset_z.npy")[0], dtype=torch.long)
box = torch.tensor([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]], dtype=torch.float32)

out, dy = model(z, pos, box=box)
print("Output NaN?", torch.isnan(out).any())
print("Force NaN?", torch.isnan(dy).any())
