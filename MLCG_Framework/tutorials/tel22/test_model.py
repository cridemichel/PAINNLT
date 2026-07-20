import sys
import torch
sys.path.append("../../training")
from models.painn import PaiNNModel

print("Loading model...")
# I need to see if the state_dict actually contains the bias keys and what the output is.
try:
    print("Loading TorchScript model...")
    model = torch.jit.load("tel22_model.pt", map_location="cpu")
    print("Model loaded successfully!")
    
    # Let's run a dummy input
    num_atoms = 10
    atomic_numbers = torch.ones(num_atoms, dtype=torch.long)
    r_ij = torch.randn(20, 3)
    edge_index = torch.randint(0, num_atoms, (2, 20))
    batch_indices = torch.zeros(num_atoms, dtype=torch.long)
    
    out = model.forward_with_rij(atomic_numbers, r_ij, edge_index, batch_indices)
    print("Model output:", out)

    
except Exception as e:
    print("Error:", e)
