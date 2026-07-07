import torch
model = torch.jit.load("../GROMACS/best_cg_model.pt")
for param in model.parameters():
    print(param.dtype)
    break
