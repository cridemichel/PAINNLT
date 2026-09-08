# TEL22 shared CGnet-style head patch

This patch transfers the representation lesson from the validated Ala2 CGnet
benchmark to TEL22 without changing the source topology or regenerating the
Variant-A residual dataset.

It adds:

- architecture `tel22_shared_geometry_tanh_v1`;
- a 131-feature orientation-aware descriptor for each of ten TEL22 copies;
- a shared, head-only tanh MLP whose ten local energies are summed;
- trainer-only normalization over the training split and all copies;
- fail-closed dataset and ESPResSo particle/edge contracts;
- a dataset-wide structural preflight that retains the physical cutoff;
- explicit augmentation of only the required TEL22 descriptor edges;
- manifest/runtime propagation of `ordered_geometry_copies`;
- a 15-epoch training-only diagnostic and machine-readable report;
- tests and scientific documentation.

The experiment deliberately reuses Morse-free Variant-A targets.  It does not
restore Morse contacts, alter the antiparallel topology, or certify a model for
production dynamics.
