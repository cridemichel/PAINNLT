import torch
import numpy as np
import argparse

# ATTENZIONE: Questo script richiede l'installazione di torchmd e torchmd-net
try:
    from torchmdnet.models.model import load_model
except ImportError:
    print("[ERRORE] Assicurati di aver installato torchmd e torchmd-net!")
    print("         pip install git+https://github.com/torchmd/torchmd.git")
    print("         pip install git+https://github.com/torchmd/torchmd-net.git")
    exit(1)

# --- Classe Custom per l'energia WCA in PyTorch ---
class WCAPrior(torch.nn.Module):
    def __init__(self, epsilon, sigma):
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = sigma * (2.0 ** (1.0 / 6.0))

    def forward(self, positions, box):
        # positions: (N, 3)
        N = positions.shape[0]
        energy = torch.tensor(0.0, device=positions.device)
        
        diff = positions.unsqueeze(1) - positions.unsqueeze(0) # (N, N, 3)
        if box is not None:
            # box: (3,)
            diff = diff - box * torch.round(diff / box)
            
        r_sq = torch.sum(diff**2, dim=-1) # (N, N)
        r_sq.fill_diagonal_(float('inf'))
        
        mask = r_sq < (self.cutoff**2)
        r_sq_masked = r_sq[mask]
        
        if len(r_sq_masked) > 0:
            sr2 = (self.sigma**2) / r_sq_masked
            sr6 = sr2**3
            sr12 = sr6**2
            e_wca = 4.0 * self.epsilon * (sr12 - sr6) + self.epsilon
            energy = energy + 0.5 * torch.sum(e_wca)

        return energy

# --- Classe Adattatore per TorchMD-Net ---
class NeuralNetworkPrior(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.model = load_model(model_path, derivative=True).to(device)
        self.model.eval()

    def forward(self, positions, atomic_numbers, box):
        # TorchMD-Net si aspetta (N, 3) posizioni e (N,) tipi atomici
        # Restituisce (Energy, Forces)
        energy, forces = self.model(atomic_numbers, positions, box=None)
        
        # Riporta le forze alla scala originale (abbiamo scalato di 0.001 nel training)
        forces = forces * 1000.0
        return energy.squeeze(), forces

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Percorso al checkpoint .ckpt (es. checkpoints/epoch=100.ckpt)")
    parser.add_argument("--dataset", required=True, help="File .npz per estrarre la scatola iniziale")
    parser.add_argument("--steps", type=int, default=1000, help="Numero di step MD")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Dispositivo: {device}")

    # 1. Carica le posizioni iniziali e i tipi
    data = np.load(args.dataset)
    pos_init = torch.tensor(data['pos'][0], dtype=torch.float32, device=device)
    z_types = torch.tensor(data['z'], dtype=torch.long, device=device)
    box = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32, device=device) # Esempio: box da 2 nm

    # 2. Inizializza le variabili MD
    natoms = len(z_types)
    pos = pos_init.clone()  # (N, 3)
    vel = torch.zeros_like(pos)
    masses = torch.ones(natoms, 1, device=device) * 18.0 
    box = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32, device=device)

    # 3. Definisci i Termini di Forza
    # - Potenziale Classico (WCA)
    wca_prior = WCAPrior(epsilon=1.0, sigma=0.3).to(device)
    # - Modello ML (TorchMD-Net)
    ml_prior = NeuralNetworkPrior(args.model, device).to(device)

    # In TorchMD, Forces calcola ad ogni step l'energia e fa l'autograd
    def force_calculator(positions, box):
        positions.requires_grad_(True)
        e_wca = wca_prior(positions, box)
        wca_forces = -torch.autograd.grad(e_wca, positions)[0]
        
        e_ml, ml_forces = ml_prior(positions, z_types, box)
        e_tot = e_wca.detach() + e_ml.detach()
        forces = wca_forces + ml_forces
        
        return e_tot, forces

    # 4. Inizializza Integratore
    # Langevin(system, forces_calculator, kT, gamma, dt)
    # NOTA: TorchMD standard usa le proprie classi Forces interne, qui simuliamo 
    # l'injection del calcolo per far capire il concetto "End-to-End".
    print("[INFO] Inizio Dinamica Molecolare su GPU...")
    
    # Loop MD base (Velocity Verlet)
    dt = 0.002
    trajectory = []
    for step in range(args.steps):
        # 1. Calcolo Forze
        energy, forces = force_calculator(pos, box)
        
        # 2. Update velocità e posizioni (Euler)
        acc = forces / masses # (N, 3) / (N, 1)
        vel = vel + 0.5 * acc * dt
        pos = pos + vel * dt
        
        if step % 10 == 0:
            print(f"Step {step:4d} | Energia: {energy.item():.4f}")
            trajectory.append(pos.detach().cpu().numpy())

    print("[INFO] Simulazione terminata!")
    np.save("trajectory.npy", np.array(trajectory))
    print("[INFO] Traiettoria salvata in trajectory.npy")

if __name__ == "__main__":
    main()
