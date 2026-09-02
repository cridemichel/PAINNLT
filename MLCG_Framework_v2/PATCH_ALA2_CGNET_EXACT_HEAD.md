# Ala2 CGnet-exact ordered-head diagnostic

This patch adds a five-epoch framework-native isolation of the dense model used
by the pinned official CGnet Ala2 comparator. It follows the corrected
`painn_ordered_geometry_tanh_v2` run, which reached `0.691%` explained
validation residual-force variance but did not materially improve on canonical
PaiNN.

The new architecture variant is `cgnet_ordered_geometry_tanh_v1`. It uses the
same converted Ala2 dataset, residual targets, harmonic priors and 8000/2000
tail split, but trains only the ordered 17-feature dense head. PaiNN embedding,
message/update blocks and readout are absent. Feature ordering and Xavier
weight initialization match the official CGnet implementation; the default
linear biases are retained. The head remains a conservative scalar-energy
model with the corrected `4.184 kJ/mol` scale.

## Apply and verify

From the framework root:

```bash
patch --dry-run -p1 < ALA2_CGNET_EXACT_HEAD.patch
patch -p1 < ALA2_CGNET_EXACT_HEAD.patch
python3 -m unittest -v tests.test_ala2_cgnet_benchmark
cmake --build training/build -j
```

Run the training-only diagnostic:

```bash
PYTHON_BIN="$(command -v python3)" \
TRAINER="$PWD/training/build/train_painn" \
ALA2_CGNET_EXACT_SOURCE_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_harmonic_50ep" \
ALA2_CGNET_EXACT_RUN_DIR="$PWD/tutorials/ala2_cgnet/diagnostics/smoke/cgnet_exact_head_5ep" \
bash tutorials/ala2_cgnet/diagnostics/scripts/06_test_ala2_cgnet_exact_head.sh
```

Inspect or share these files from the fresh run directory:

- `ala2_benchmark_report.json`
- `cg_training_log.csv`
- `training_stdout.log`

Do not run the optional FES screen until the five-epoch report has been
compared with the canonical PaiNN (`~0.65%`) and official CGnet (`~0.86%`)
force-skill controls.

The ESPResSo bridge must guard its PaiNN-specific isolated-species diagnostic:
the CGnet-exact checkpoint has no embedding or PaiNN readout by construction.
For head-only inference it reports the ordered-head zero-feature gauge instead
of dereferencing those absent modules. This diagnostic branch does not change
energies, forces or checkpoint parameters.
