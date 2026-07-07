import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from schnetpack.data import AtomsDataModule
import schnetpack.transform as trn
import schnetpack as spk

data_module = AtomsDataModule(
    'cg_dataset.db',
    batch_size=2,
    num_train=800,
    num_val=200,
    transforms=[trn.ASENeighborList(cutoff=1.0), trn.CastTo32()],
    num_workers=0
)
data_module.setup()
print("Setup completato")
dl = data_module.train_dataloader()
batch = next(iter(dl))
print("Batch ottenuto")

# Model
n_features = 64
schnet = spk.representation.SchNet(n_atom_basis=n_features, n_interactions=2, radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=1.0), cutoff_fn=spk.nn.CosineCutoff(1.0))
pred_energy = spk.atomistic.Atomwise(n_in=n_features, output_key='energy')
pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[spk.atomistic.PairwiseDistances()],
    output_modules=[pred_energy, pred_forces],
)
print("Modello creato")

print("Eseguo forward pass...", flush=True)
with torch.no_grad():
    res = nnpot(batch)
print("Forward completato:", list(res.keys()))
