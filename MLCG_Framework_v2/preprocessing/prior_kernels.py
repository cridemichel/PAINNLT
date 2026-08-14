"""Vectorized conservative prior kernels used during dataset construction."""

from __future__ import annotations

import numpy as np


def switched_morse_radial_force_array(r, D, a, r0, r_switch, r_cut):
    """Return signed radial switched-Morse forces for NumPy arrays.

    This is the preprocessing counterpart of the analytic switched Morse used
    by the ESPResSo runtime.  The sign convention is radial force along the
    vector from the second particle to the first particle.
    """
    r = np.asarray(r, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64)
    r_switch = np.asarray(r_switch, dtype=np.float64)
    r_cut = np.asarray(r_cut, dtype=np.float64)
    force = np.zeros_like(r)
    active = r < r_cut
    if not np.any(active):
        return force

    rr = r[active]
    DD = D[active]
    aa = a[active]
    rr0 = r0[active]
    rsw = r_switch[active]
    rcut = r_cut[active]
    y = np.exp(-aa * (rr - rr0))
    base_energy = DD * (y * y - 2.0 * y)
    base_force = 2.0 * DD * aa * y * (y - 1.0)
    local_force = base_force.copy()

    switched = rr > rsw
    if np.any(switched):
        width = rcut[switched] - rsw[switched]
        t = (rr[switched] - rsw[switched]) / width
        switch = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
        d_switch_dr = -30.0 * t * t * (1.0 - t) * (1.0 - t) / width
        local_force[switched] = (
            switch * base_force[switched]
            - base_energy[switched] * d_switch_dr
        )

    force[active] = local_force
    return force
