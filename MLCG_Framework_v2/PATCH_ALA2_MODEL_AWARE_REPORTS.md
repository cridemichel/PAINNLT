# Ala2 model-aware report correction

Apply from the framework root after the CGnet-exact-head patch and its runtime
gauge hotfix:

```bash
patch --dry-run -p1 < ALA2_MODEL_AWARE_REPORTS.patch
patch -p1 < ALA2_MODEL_AWARE_REPORTS.patch
python3 -m unittest -v tests.test_ala2_cgnet_benchmark
```

The patch changes report vocabulary only. It does not alter training,
potentials, checkpoints, priors, integration parameters or stored scientific
results.

New FES runs write `prior_plus_model_samples.npz`; the official comparator
continues to accept the schema-v1 `prior_plus_painn_samples.npz` filename.
New reports use schema version 2 and identify the concrete architecture in
`model_identity`.

The validated Ala2 numbers and the constraints for the subsequent TEL22
transfer are recorded in
`tutorials/ala2_cgnet/CGNET_EXACT_VALIDATION.md`.
