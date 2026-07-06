import torch
import numpy as np
import struct
import json

torch.manual_seed(42)

# Load model
model = torch.jit.load("GROMACS/best_cg_model_with_priors.pt", map_location='cpu')
with open("GROMACS/best_cg_model_with_priors_config.json") as f:
    config = json.load(f)
cutoff = float(config['cutoff'])

# Load 9 molecules
with open("GROMACS/cg_dataset_priors.bin", "rb") as f:
    f.read(4)
    num_molecules, num_total_sites = struct.unpack("ii", f.read(8))
    box = torch.tensor(struct.unpack("3f", f.read(12)), dtype=torch.float32)
    pos = []
    for mol_idx in range(9):
        mol_id, num_sites = struct.unpack("ii", f.read(8))
        cx, cy, cz, fx, fy, fz, tx, ty, tz = struct.unpack("9f", f.read(36))
        pos.append([cx, cy, cz])
        for s in range(num_sites): f.read(16)

pos = torch.tensor(pos, dtype=torch.float32) % box
box_diag = torch.diag(box)
z = torch.ones(9, dtype=torch.long)

mass = 18.015 # water mass
mass_tensor = torch.full((9, 1), mass, dtype=torch.float32)

def run_pure_nve(dt_ps, precision):
    if precision == torch.float64:
        # Purtroppo TorchScript model potrebbe forzare float32 internamente, ma ci proviamo
        _pos = pos.to(torch.float64)
        _box = box_diag.to(torch.float64)
        _mass = mass_tensor.to(torch.float64)
    else:
        _pos = pos.clone()
        _box = box_diag.clone()
        _mass = mass_tensor.clone()
        
    _pos.requires_grad_(True)
    vel = torch.randn((9, 3), dtype=_pos.dtype) * 0.1
    
    t_tot = 0.1
    steps = max(10, int(round(t_tot / dt_ps)))
    
    e_tots = []
    for step in range(steps):
        # Velocity Verlet
        # 1. Update v half step
        out = model(z, _pos.to(torch.float32), _box.to(torch.float32))
        e_pot = out[0] if isinstance(out, tuple) else out
        forces = torch.autograd.grad(e_pot.sum(), _pos, retain_graph=False)[0]
        # if precision is float64, we need to cast forces
        forces = forces.to(_pos.dtype)
        
        acc = forces / _mass
        vel = vel + 0.5 * dt_ps * acc
        
        # 2. Update pos full step
        _pos = _pos + dt_ps * vel
        _pos = _pos % torch.diag(_box)
        
        # 3. Update v half step
        out = model(z, _pos.to(torch.float32), _box.to(torch.float32))
        e_pot = out[0] if isinstance(out, tuple) else out
        forces = torch.autograd.grad(e_pot.sum(), _pos, retain_graph=False)[0].to(_pos.dtype)
        acc = forces / _mass
        vel = vel + 0.5 * dt_ps * acc
        
        e_kin = 0.5 * torch.sum(_mass * vel**2)
        e_tot = e_pot.item() + e_kin.item()
        e_tots.append(e_tot)
        
    return np.std(e_tots)

print("Float32 pure PyTorch integrator:")
for dt in [0.0001, 0.001, 0.002, 0.004, 0.006]:
    dE = run_pure_nve(dt, torch.float32)
    print(f"dt = {dt*1000:.1f} fs -> dE = {dE:.6f}")

