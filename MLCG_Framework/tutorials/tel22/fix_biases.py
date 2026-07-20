import torch

model = torch.jit.load("tel22_model.pt")

for name, module in model.named_modules():
    if "filter_mlp" in name and isinstance(module, torch.jit.RecursiveScriptModule) and module.original_name == "Linear":
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()

model.save("tel22_model_fixed.pt")
print("Biases zeroed and model saved as tel22_model_fixed.pt")
