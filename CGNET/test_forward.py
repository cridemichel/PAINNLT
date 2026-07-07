import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
import schnetpack as spk
from schnetpack.data import collate_aseatoms

dataset = ASEAtomsData('cg_dataset.db', transforms=[trn.ASENeighborList(cutoff=1.0), trn.CastTo32()])
print("Dataset caricato")

item = dataset[0]
print("Item estratto:", list(item.keys()))

batch = collate_aseatoms([item])
print("Batch creato:", list(batch.keys()))

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

with torch.no_grad():
    res = nnpot(batch)
print("Forward completato:", list(res.keys()))
