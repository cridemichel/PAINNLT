# Ala2 official-CGnet comparator patch

This patch adds the controlled experiment that separates a PaiNN/framework
problem from the limitations of the public 10,000-frame Ala2 subset.

It does **not** replace or modify the validated TEL22 pipeline. It adds a third
Ala2 diagnostic which:

1. downloads the official `coarse-graining/cgnet` source at the pinned commit
   already used for the Ala2 arrays and verifies the archive SHA-256;
2. trains the official dense Ala2 CGnet architecture on frames 0--7999 and
   validates on frames 8000--9999;
3. runs the official overdamped Langevin integrator with the same number of
   replicas and retained samples as the completed prior/PaiNN A/B test;
4. compares prior-only, prior+PaiNN and official CGnet against the same
   atomistic Ramachandran reference, including paired bootstrap intervals.

Apply from the framework root:

```bash
patch --dry-run -p1 < ALA2_OFFICIAL_CGNET_COMPARATOR.patch
patch -p1 < ALA2_OFFICIAL_CGNET_COMPARATOR.patch
```

Run the fast unit/static checks:

```bash
python3 -m unittest discover -s tests -p 'test_ala2_cgnet_benchmark.py' -v
```

Then use the command documented in `tutorials/ala2_cgnet/README.md`, pointing
`ALA2_AB_RUN_DIR` to the already completed four-replica A/B directory. The
comparator infers the retained sample count from that directory; no PaiNN or
prior simulation is repeated.

The active Python environment must provide NumPy, SciPy, PyTorch and
Matplotlib. Network access is needed once to download the pinned official
source archive. The archive is not installed globally and its source is not
edited.
