import torch

# Load TorchScript model
model = torch.jit.load("tel22_model_fixed.pt", map_location="cpu")

# Print the names and shapes of parameters
for name, param in model.named_parameters():
    print(name, param.shape)
