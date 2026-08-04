#!/usr/bin/env python3
"""Compatibility entry point.

Tail extrapolation is now performed inside ``ibi/run_ibi_loop.py`` at every DBI
and IBI iteration.  Post-processing the generated tables would break the
energy/force consistency and must not be used.
"""

raise SystemExit(
    "Extrapolation is integrated into run_ibi_loop.py. "
    "Do not post-process TEL22 IBI tables; configure ibi_extrapolation_config.json instead."
)
