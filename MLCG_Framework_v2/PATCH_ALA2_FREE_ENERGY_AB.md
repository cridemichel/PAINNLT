# Ala2 matched free-energy A/B diagnostic

This patch extends the public Ala2 CGnet training benchmark with a physical
test of the learned potential.  It is intended to be applied after
`TEL22_ALA2_CGNET_BENCHMARK.patch`.

It adds:

- ESPResSo-compatible one-site rigid-body templates and WCA exclusion policy
  v3;
- a 0.22 nm CGnet-style excluded-volume WCA which is zero on every official
  training frame and therefore does not invalidate the completed model;
- conversion of older benchmark priors to runtime-safe, site-level bonds and
  angles without changing their energy or the residual targets used in the
  completed training;
- evenly spaced starting frames and matched prior-only/prior+PaiNN replicas;
- two-dimensional phi/psi distributions and free-energy surfaces;
- Jensen-Shannon, coverage and shifted FES-MSE metrics with a paired-replica
  bootstrap;
- optional ingestion of external CGnet samples for a direct three-model
  comparison.

The default run is deliberately smaller than the paper's protocol and reports
an `inconclusive_at_current_sampling` verdict when the paired bootstrap does
not resolve the sign of the improvement.  A scientific result is never turned
into a pipeline failure merely because PaiNN does not improve the surface.
