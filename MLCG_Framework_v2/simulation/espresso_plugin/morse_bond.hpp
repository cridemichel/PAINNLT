#pragma once

#include <utils/Vector.hpp>

#include <cmath>
#include <optional>
#include <stdexcept>

/**
 * Conservative pairwise Morse bond used by MLCG priors.
 *
 * Potential convention:
 *   U(r) = D * (1 - exp(-a * (r - r0)))^2
 */
struct MorseBond {
  double D;
  double a;
  double r0;
  double r_cut;

  static constexpr int num = 1;

  MorseBond(double D, double a, double r0, double r_cut)
      : D{D}, a{a}, r0{r0}, r_cut{r_cut} {
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
  }

  double cutoff() const { return r_cut; }

  std::optional<Utils::Vector3d>
  force(Utils::Vector3d const &dx) const {
    auto const r = dx.norm();
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
    if (r >= r_cut || r == 0.0) {
      return std::nullopt;
    }

    auto const exp_term = std::exp(-a * (r - r0));
    auto const one_minus_exp = 1.0 - exp_term;
    return D * one_minus_exp * one_minus_exp;
  }
};
