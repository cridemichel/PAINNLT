# TEL22_IBI historical archive manifest

- executed: **True**
- GROMACS generated artifacts preserved: **yes**
- live historical dependencies preserved at tutorial root: **yes**

| category | artifact | status | files | bytes | tree SHA256 |
|---|---|---|---:|---:|---|
| historical_ibi | `ibi_dbi_preview` | archived_this_run | 12 | 2090821 | `dd8f738bc1f4e39c4f7b8d113cfac582a72d057cfffadac592aee586fc29b87b` |
| historical_ibi | `ibi_run` | archived_this_run | 114 | 113551402 | `99522aa045397a55e01ae978343b9c1d24978ade7734b246e2c5859fb6bd4de2` |
| ml_residual_experiments | `training_multiseed_benchmark` | archived_this_run | 27 | 2964693 | `cc50867f58a88c2b0903b6d1f27b1b203e1c63dc3834559f89a1115fdbf8a03f` |
| ml_residual_experiments | `ibi_ml_ab_validation` | archived_this_run | 12 | 7918388 | `f23d86c8a5a0e722f459f71697da0e0efd9438b9550af735a4c795f42eb24482` |

## Preserved live historical dependencies

- `ibi_run_16ps`
- `ibi_run_16ps_continue`
- `ibi_validation_best`
- `postibi_runtime_validation`
- `tel22_dataset_ibi_residual.bin`
- `tel22_model_ibi.pt`
- `tel22_model_ibi.pt.manifest.json`
- `tel22_model_ibi_conservative.pt`
- `tel22_model_ibi_conservative.pt.manifest.json`
- `ibi_residual_build_manifest.json`

## GROMACS preservation policy

The archive operation does not move or remove AA/GROMACS products. This includes
`md.trr`, `md_whole.trr`, `md.gro`, TPR/CPT/EDR/LOG files, EM/NVT/NPT outputs,
solvated/ionized GRO files, topology/position-restraint files, and source PDB files.
