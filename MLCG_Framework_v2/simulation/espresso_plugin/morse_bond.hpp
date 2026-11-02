#pragma once

#include <utils/Vector.hpp>

#include <cmath>
#include <optional>
#include <stdexcept>

/**
 * Pairwise Morse bond used by MLCG priors.
 *
 * Two conventions, selected by ``switch_start``:
 *
 * - ``switch_start < 0`` (default, legacy analytic diagnostic):
 *     U(r) = D * (1 - exp(-a * (r - r0)))^2   for 0 < r < r_cut;
 *   at r >= r_cut (or r == 0) the kernel returns no value, which ESPResSo
 *   treats as a broken bond.
 *
 * - ``switch_start >= 0`` (reversible switched contact, 2026-11): exactly the
 *   production pair-specific contact of the MLCG switched non-bonded Morse
 *   (morse_switched.hpp, physical gauge U(r0) = -D, U(inf) = 0),
 *     U(r) = D (y^2 - 2 y),  y = exp(-a (r - r0)),            r <= switch_start
 *     U(r) = s(t) D (y^2 - 2 y),  t = (r - switch_start) / (r_cut - switch_start),
 *            s = 1 - 10 t^3 + 15 t^4 - 6 t^5,                 switch_start < r < r_cut
 *     U(r) = 0, F = 0                                          r >= r_cut
 *   The contact can break and re-form: beyond r_cut the bond is inert, never
 *   broken.  A bond acts on two given particles, so a contact costs no
 *   particle type: the number of types does not grow with the number of
 *   copies (the marker-nonbonded realisation needs one type per endpoint and
 *   per copy, and ESPResSo stores parameters for every pair of types).
 */
struct MorseBond {
  double D;
  double a;
  double r0;
  double r_cut;
  double switch_start;

  static constexpr int num = 1;

  MorseBond(double D, double a, double r0, double r_cut,
            double switch_start = -1.0)
      : D{D}, a{a}, r0{r0}, r_cut{r_cut}, switch_start{switch_start} {
    if (D < 0.0) {
      throw std::domain_error("Parameter 'D' must be >= 0");
    }
    if (a <= 0.0) {
      throw std::domain_error("Parameter 'a' must be > 0");
    }
    if (r0 < 0.0) {
      throw std::domain_error("Parameter 'r_0' must be >= 0");
    }
    if (r_cut <= 0.0) {
      throw std::domain_error("Parameter 'r_cut' must be > 0");
    }
    if (switch_start >= 0.0 && !(r0 < switch_start && switch_start < r_cut)) {
      throw std::domain_error(
          "Switched MorseBond requires r_0 < switch_start < r_cut");
    }
  }

  // Switched mode: the contact must not set the interaction range of the cell
  // system.  r_cut is ~15 nm by default (negligible tail), larger than half the
  // box of the 10-copy systems, and the cell system does not need it: on one
  // MPI rank ESPResSo always finds the partner of a bond among the local
  // particles and uses the minimum-image distance.  run_cg_md.py refuses
  // bonded-switched on more than one rank.
  double cutoff() const { return switch_start >= 0.0 ? 0.0 : r_cut; }

  bool switched() const { return switch_start >= 0.0; }

  std::optional<Utils::Vector3d>
  force(Utils::Vector3d const &dx) const {
    auto const r = dx.norm();
    if (switched()) {
      if (r >= r_cut || r == 0.0) {
        return 0.0 * dx;
      }
      auto const y = std::exp(-a * (r - r0));
      auto const base_force = 2.0 * D * a * y * (y - 1.0);   // -dU/dr
      if (r <= switch_start) {
        return (base_force / r) * dx;
      }
      auto const base_energy = D * (y * y - 2.0 * y);
      auto const width = r_cut - switch_start;
      auto const t = (r - switch_start) / width;
      auto const t2 = t * t;
      auto const t3 = t2 * t;
      auto const sw = 1.0 - 10.0 * t3 + 15.0 * t3 * t - 6.0 * t3 * t2;
      auto const dsw_dr = -30.0 * t2 * (1.0 - t) * (1.0 - t) / width;
      auto const radial_force = sw * base_force - base_energy * dsw_dr;
      return (radial_force / r) * dx;
    }
    if (r >= r_cut || r == 0.0) {
      return std::nullopt;
    }

    auto const exp_term = std::exp(-a * (r - r0));
    auto const radial_force =
        -2.0 * a * D * (1.0 - exp_term) * exp_term;
    return (radial_force / r) * dx;
  }

  std::optional<double> energy(Utils::Vector3d const &dx) const {
    auto const r = dx.norm();
    if (switched()) {
      if (r >= r_cut || r == 0.0) {
        return 0.0;
      }
      auto const y = std::exp(-a * (r - r0));
      auto const base_energy = D * (y * y - 2.0 * y);
      if (r <= switch_start) {
        return base_energy;
      }
      auto const t = (r - switch_start) / (r_cut - switch_start);
      auto const t3 = t * t * t;
      auto const sw = 1.0 - 10.0 * t3 + 15.0 * t3 * t - 6.0 * t3 * t * t;
      return sw * base_energy;
    }
    if (r >= r_cut || r == 0.0) {
      return std::nullopt;
    }

    auto const exp_term = std::exp(-a * (r - r0));
    auto const one_minus_exp = 1.0 - exp_term;
    return D * one_minus_exp * one_minus_exp;
  }
};
