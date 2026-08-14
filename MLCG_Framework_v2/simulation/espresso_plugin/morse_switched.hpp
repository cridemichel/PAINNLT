/*
 * MLCG extension of ESPResSo's non-bonded Morse kernel.
 *
 * Stock mode (switch_start < 0) is bit-for-bit algebraically equivalent to
 * ESPResSo's shifted Morse interaction.  Switched mode multiplies the
 * unshifted Morse potential by a quintic switching function so that both
 * energy and force vanish continuously at cutoff.
 */
#pragma once

#include <config/config.hpp>

#ifdef ESPRESSO_MORSE

#include "nonbonded_interaction_data.hpp"

#include <cmath>

inline double morse_pair_force_factor(IA_parameters const &ia_params,
                                      double dist) {
  auto const &p = ia_params.morse;
  if (dist >= p.cut) {
    return 0.0;
  }

  auto const y = std::exp(-p.alpha * (dist - p.rmin));
  auto const base_force = 2.0 * p.eps * p.alpha * y * (y - 1.0);

  // Negative switch_start selects ESPResSo's original shifted Morse.
  // The constant energy shift does not affect this force expression.
  if (p.switch_start < 0.0 || dist <= p.switch_start) {
    return base_force / dist;
  }

  auto const base_energy = p.eps * (y * y - 2.0 * y);
  auto const width = p.cut - p.switch_start;
  auto const t = (dist - p.switch_start) / width;
  auto const t2 = t * t;
  auto const t3 = t2 * t;
  auto const t4 = t3 * t;
  auto const t5 = t4 * t;
  auto const sw = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5;
  auto const dsw_dr = -30.0 * t2 * (1.0 - t) * (1.0 - t) / width;
  auto const radial_force = sw * base_force - base_energy * dsw_dr;
  return radial_force / dist;
}

inline double morse_pair_energy(IA_parameters const &ia_params, double dist) {
  auto const &p = ia_params.morse;
  if (dist >= p.cut) {
    return 0.0;
  }

  auto const y = std::exp(-p.alpha * (dist - p.rmin));
  auto const base_energy = p.eps * (y * y - 2.0 * y);

  if (p.switch_start < 0.0) {
    // ESPResSo stock convention: shift the energy so U(cutoff)=0.
    return base_energy - p.rest;
  }
  if (dist <= p.switch_start) {
    // Physical non-covalent gauge: U(rmin)=-eps and U(infinity)=0.
    return base_energy;
  }

  auto const width = p.cut - p.switch_start;
  auto const t = (dist - p.switch_start) / width;
  auto const t2 = t * t;
  auto const t3 = t2 * t;
  auto const t4 = t3 * t;
  auto const t5 = t4 * t;
  auto const sw = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5;
  return sw * base_energy;
}

#endif // ESPRESSO_MORSE
