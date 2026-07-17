import torch

model = torch.jit.load("tel22_model.pt")

class CappedModel(torch.nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.orig = orig

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        forces = self.orig(z, pos, batch)
        return torch.clamp(forces, min=-1000.0, max=1000.0)

capped_model = CappedModel(model)

z = torch.ones(22, dtype=torch.long)
pos = torch.rand(22, 3)
batch = torch.zeros(22, dtype=torch.long)

traced = torch.jit.trace(capped_model, (z, pos, batch))
traced.save("tel22_model_capped.pt")
print("Successfully created tel22_model_capped.pt")
