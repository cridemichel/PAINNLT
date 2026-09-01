# Ala2 CGnet benchmark patch

This patch adds a force-only PaiNN diagnostic based on the official public
alanine-dipeptide arrays from `coarse-graining/cgnet`.

Apply from the framework root:

```bash
patch --dry-run -p1 < TEL22_ALA2_CGNET_BENCHMARK.patch
patch -p1 < TEL22_ALA2_CGNET_BENCHMARK.patch
```

Run the local unit tests first:

```bash
python3 -m unittest -v tests/test_ala2_cgnet_benchmark.py
```

Then run the benchmark:

```bash
PYTHON_BIN="$(command -v python3)" \
TRAINER="$PWD/training/build/train_painn" \
bash tutorials/ala2_cgnet/diagnostics/scripts/01_test_ala2_cgnet_benchmark.sh
```

The benchmark does not modify TEL22 sources or results.
