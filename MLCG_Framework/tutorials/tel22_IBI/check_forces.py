import torch
import numpy as np
import json
import sys
import os
sys.path.append(os.path.abspath("../../training/src"))
import dataset
from model import PaiNN

print("Loading dataset...")
ds = dataset.CGDataset("tel22_dataset_ibi.bin")
print(f"Dataset size: {len(ds)}")

with open("tel22_training_config.json") as f:
    config = json.load(f)

model = PaiNN(
    num_species=config["num_species"],
    hidden_channels=config["hidden_channels"],
    n_layers=config["n_layers"],
    num_rbf=config["num_rbf"],
    cutoff=config["cutoff"]
)
model.load_state_dict(torch.load("tel22_model_ibi.pt", map_location="cpu"))
model.eval()

data = ds[0]
pos = data.pos.unsqueeze(0)
species = data.species.unsqueeze(0)

# Evaluate model
with torch.no_grad():
    ml_forces, ml_torques = model(pos, species)

target_f = data.force
target_t = data.torque

print("ML Forces (first 5):")
print(ml_forces[0, :5])
print("Target Forces (first 5):")
print(target_f[:5])

error_f = torch.abs(ml_forces.squeeze(0) - target_f)
print(f"Max Force Error: {error_f.max().item():.2f}")
print(f"Mean Force Error: {error_f.mean().item():.2f}")

print("Max ML Force:", ml_forces.abs().max().item())
print("Max Target Force:", target_f.abs().max().item())
