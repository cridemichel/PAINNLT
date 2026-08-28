# TEL22 antiparallel-topology patch

This patch corrects the pair-specific Morse graph for the antiparallel basket
fold in `MODEL 1` of PDB 143D.

## Correct tetrads

One-based residue numbers within each 22-residue TEL22 copy:

```text
2  10  14  22
3   9  15  21
4   8  16  20
```

Each tetrad is encoded as a complete K4 graph. The graph is repeated for ten
copies with an offset of 22 residues. All endpoints remain explicitly COM-COM
(`site_i = site_j = -1`). The Morse depth and width remain `D = 50 kJ/mol` and
`a = 0.3 nm^-1`.

Every corrected Morse record has `r0 = "auto"`. Old numeric values belonged to
different residue pairs and must not be transferred. Consequently, all old
generated `cg_priors.json`, `tel22_dataset.bin`, models and checkpoints are
incompatible with this patch.

## Static test

```bash
python3 tutorials/tel22/diagnostics/scripts/validate_antiparallel_topology.py \
  --topology tutorials/tel22/tel22_topology.json \
  --pdb tutorials/tel22/143D.pdb \
  --r0-mode auto \
  --require-reference-metadata
```

## Local end-to-end test

```bash
AA_TOPOLOGY=/path/to/md.gro \
AA_TRAJECTORY=/path/to/md_whole.trr \
TRAINER=/path/to/training/build/train_painn \
PYRESSO=/path/to/espresso/build/pypresso \
bash tutorials/tel22/diagnostics/scripts/07_test_antiparallel_pipeline_40ep.sh
```

The runner uses a fresh directory under `tutorials/tel22/diagnostics/smoke/`,
trains for exactly 40 epochs, performs short equilibration and NVT stages, and
writes `pipeline_test_report.json`. This is a functional test, not an NVE or
thermodynamic certification.
