import torch
import schnetpack as spk
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from schnetpack.data import ASEAtomsData
from schnetpack.data.loader import _atoms_collate_fn
import schnetpack.transform as trn

cutoff = 1.0
ds = ASEAtomsData('cg_dataset.db')
item = ds[0]
nl = trn.TorchNeighborList(cutoff=cutoff)
item = nl(item)
cast = trn.CastTo32()
item = cast(item)

batch = _atoms_collate_fn([item])
batch['_positions'].requires_grad_(True)

schnet = spk.representation.SchNet(
    n_atom_basis=64, 
    n_interactions=2,
    radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)
pairwise_distance = spk.atomistic.PairwiseDistances()
pred_energy = spk.atomistic.Atomwise(n_in=64, output_key='energy')
pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces]
)

res = nnpot(batch)
print("Target forces:", batch['forces'].abs().max())
print("Pred forces:", res['forces'].abs().max())
