import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule
import sys

def trace_calls(frame, event, arg):
    if event == 'call':
        func_name = frame.f_code.co_name
        if func_name in ['exit', 'sys.exit', '_exit']:
            print(f"EXIT CALLED: {func_name} in {frame.f_code.co_filename}:{frame.f_lineno}")
    return trace_calls
# sys.settrace(trace_calls)

try:
    print("Creating datamodule")
    data_module = AtomsDataModule(
        'cg_dataset.db',
        batch_size=8,
        num_train=800,
        num_val=200,
        transforms=[trn.ASENeighborList(cutoff=1.0), trn.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False), trn.CastTo32()],
        num_workers=0, num_val_workers=0, num_test_workers=0, pin_memory=False
    )
    print("Calling setup")
    data_module.setup()
    print("Setup done")
except Exception as e:
    print("Exception", e)
