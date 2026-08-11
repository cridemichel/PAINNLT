TEL22 force-mapping / atomistic force-source audit
==================================================

Main result:
  force_mapping_source_report.json

Interpretation rule:
  Read source magnitudes ONLY if closure.status == PASS.
  The source decomposition is produced by exact GROMACS subset reruns:
    DNA-only          -> DNA-DNA contribution
    DNA+water - DNA   -> water-on-DNA contribution
    DNA+K - DNA       -> K-on-DNA contribution
    DNA+Cl - DNA      -> Cl-on-DNA contribution

The script also checks that the current per-residue COM coordinate mapping and
simple force sum satisfy B C^T = I for translations.

IMPORTANT: residual_minus_environment is a diagnostic quantity only. It is not
recommended as a new training target without a separate thermodynamic argument.
