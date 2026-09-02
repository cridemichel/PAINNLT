# Ala2 matched Brownian prior/CGnet A/B patch

This incremental patch applies after `ALA2_OFFICIAL_CGNET_COMPARATOR.patch`.
It removes the main confounder from the first comparison by adding a second
official-CGnet simulation branch with the neural energy disabled.

The two Brownian branches use the same:

- fitted harmonic bonds and angles;
- initial reference frames;
- overdamped integrator and time step;
- trajectory length, burn-in and sampling interval;
- reset random seed and therefore the same Brownian noise sequence throughout
  both paths.

The analysis adds a five-panel FES figure and the JSON section
`cgnet_external.matched_brownian_ab`, with aggregate metrics, per-replica
metrics, a paired replica bootstrap and an explicit scientific verdict.

Apply from the framework root:

```bash
patch --dry-run -p1 < ALA2_CGNET_BROWNIAN_AB.patch
patch -p1 < ALA2_CGNET_BROWNIAN_AB.patch
```

Run the unit/static checks:

```bash
python3 -m unittest discover -s tests -p 'test_ala2_cgnet_benchmark.py' -v
```

Use a fresh `ALA2_CGNET_COMPARATOR_RUN_DIR`; the previous
`official_cgnet_quick` directory is intentionally never overwritten. The run
reuses the completed ESPResSo A/B samples and takes approximately twice the
simulation time of the first CGnet comparator.
