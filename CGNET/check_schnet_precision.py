import torch
model = torch.load("schnet_model_dir/best_model", map_location='cpu', weights_only=False)
for param in model.parameters():
    print(param.dtype)
    break
