import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule

print("Start", flush=True)
data_module = AtomsDataModule(
    'cg_dataset.db',
    batch_size=8,
    num_train=800,
    num_val=200,
    transforms=[trn.ASENeighborList(cutoff=1.0), trn.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False), trn.CastTo32()],
    num_workers=0, num_val_workers=0, num_test_workers=0, pin_memory=False
)
data_module.setup()
print("Setup complete", flush=True)
dl = data_module.train_dataloader()
print("Dataloader created", flush=True)
for batch in dl:
    print("Batch loaded!", flush=True)
    break
print("Success", flush=True)
