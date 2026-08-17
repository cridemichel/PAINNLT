/*
 * MLCG Framework v2 conservative bonded spline for ESPResSo 5.0.x.
 *
 * One scalar cubic-Hermite energy interpolant U(q) is the source of truth.
 * Energy and dU/dq are evaluated from the same polynomial, eliminating the
 * independent energy/force interpolation used by Tabulated* interactions.
 */
#pragma once

#include "angle_common.hpp"
#include "dihedral.hpp"

#include <utils/Vector.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

struct ConservativeSpline1D {
  double minval = 0.0;
  double maxval = 0.0;
  double inv_step = 0.0;
  std::vector<double> energy_nodes;
  std::vector<double> derivative_nodes;
  bool periodic = false;

  ConservativeSpline1D() = default;

  ConservativeSpline1D(double min, double max,
                       std::vector<double> const &energy,
                       std::vector<double> const &derivative,
                       bool periodic_input = false)
      : minval(min), maxval(max), energy_nodes(energy),
        derivative_nodes(derivative), periodic(periodic_input) {
    if (!(maxval > minval)) {
      throw std::invalid_argument("Conservative spline requires max > min");
    }
    if (energy_nodes.size() < 2 || energy_nodes.size() != derivative_nodes.size()) {
      throw std::invalid_argument(
          "Conservative spline energy/derivative arrays must have equal length >= 2");
    }
    for (auto const value : energy_nodes) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Conservative spline energy contains NaN/Inf");
      }
    }
    for (auto const value : derivative_nodes) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Conservative spline derivative contains NaN/Inf");
      }
    }
    if (periodic) {
      auto const escale = std::max({1.0, std::abs(energy_nodes.front()),
                                    std::abs(energy_nodes.back())});
      auto const dscale = std::max({1.0, std::abs(derivative_nodes.front()),
                                    std::abs(derivative_nodes.back())});
      if (std::abs(energy_nodes.front() - energy_nodes.back()) > 1.0e-10 * escale ||
          std::abs(derivative_nodes.front() - derivative_nodes.back()) > 1.0e-10 * dscale) {
        throw std::invalid_argument(
            "Periodic conservative spline requires matching endpoint U and dU/dq");
      }
    }
    inv_step = static_cast<double>(energy_nodes.size() - 1) / (maxval - minval);
  }

  double cutoff() const { return maxval; }

  std::pair<double, double> evaluate(double q) const {
    if (periodic) {
      auto const period = maxval - minval;
      q = minval + std::fmod(q - minval, period);
      if (q < minval) {
        q += period;
      }
      // maxval is the seam duplicate of minval and is never evaluated as a
      // separate physical point.  This makes U and dU/dq exactly periodic.
      if (q >= maxval) {
        q = minval;
      }
    }

    // Conservative linear tangent continuation below the first node.
    if (!periodic && q < minval) {
      auto const dq = q - minval;
      return {energy_nodes.front() + derivative_nodes.front() * dq,
              derivative_nodes.front()};
    }

    auto const n = energy_nodes.size();
    double scaled = (q - minval) * inv_step;
    std::size_t i = 0;
    double t = 0.0;
    if (q >= maxval) {
      i = n - 2;
      t = 1.0;
    } else {
      scaled = std::max(0.0, scaled);
      i = std::min(static_cast<std::size_t>(std::floor(scaled)), n - 2);
      t = scaled - static_cast<double>(i);
    }

    auto const h = 1.0 / inv_step;
    auto const y0 = energy_nodes[i];
    auto const y1 = energy_nodes[i + 1];
    auto const m0 = derivative_nodes[i];
    auto const m1 = derivative_nodes[i + 1];
    auto const t2 = t * t;
    auto const t3 = t2 * t;

    auto const h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    auto const h10 = t3 - 2.0 * t2 + t;
    auto const h01 = -2.0 * t3 + 3.0 * t2;
    auto const h11 = t3 - t2;
    auto const energy = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;

    auto const dh00 = 6.0 * t2 - 6.0 * t;
    auto const dh10 = 3.0 * t2 - 4.0 * t + 1.0;
    auto const dh01 = -6.0 * t2 + 6.0 * t;
    auto const dh11 = 3.0 * t2 - 2.0 * t;
    auto const derivative =
        (dh00 * y0 + dh01 * y1) / h + dh10 * m0 + dh11 * m1;
    return {energy, derivative};
  }
};

struct ConservativeSplineDistanceBond {
  static constexpr int num = 1;
  ConservativeSpline1D spline;

  ConservativeSplineDistanceBond(double min, double max,
                                 std::vector<double> const &energy,
                                 std::vector<double> const &derivative)
      : spline(min, max, energy, derivative) {}

  double cutoff() const { return spline.cutoff(); }

  std::optional<Utils::Vector3d> force(Utils::Vector3d const &dx) const {
    auto const dist = dx.norm();
    if (dist < spline.cutoff()) {
      if (dist <= 0.0) {
        throw std::domain_error("ConservativeSplineDistance undefined at zero distance");
      }
      auto const dU_dr = spline.evaluate(dist).second;
      auto const radial_force = -dU_dr;
      return (radial_force / dist) * dx;
    }
    return {};
  }

  std::optional<double> energy(Utils::Vector3d const &dx) const {
    auto const dist = dx.norm();
    if (dist < spline.cutoff()) {
      return spline.evaluate(dist).first;
    }
    return {};
  }
};

struct ConservativeSplineAngleBond {
  static constexpr int num = 2;
  ConservativeSpline1D spline;

  ConservativeSplineAngleBond(double min, double max,
                              std::vector<double> const &energy,
                              std::vector<double> const &derivative)
      : spline(min, max, energy, derivative) {
    if (std::abs(min) > 1.0e-12 || std::abs(max - std::acos(-1.0)) > 1.0e-10) {
      throw std::invalid_argument("ConservativeSplineAngle requires range 0..pi");
    }
  }

  double cutoff() const { return 0.0; }

  std::tuple<Utils::Vector3d, Utils::Vector3d, Utils::Vector3d>
  forces(Utils::Vector3d const &vec1, Utils::Vector3d const &vec2) const {
    auto force_factor = [this](double const cos_phi) {
      auto const sin_phi = std::sqrt(std::max(0.0, 1.0 - cos_phi * cos_phi));
      auto const phi = std::acos(cos_phi);
      auto const dU_dphi = spline.evaluate(phi).second;
      return -dU_dphi / sin_phi;
    };
    return angle_generic_force(vec1, vec2, force_factor, true);
  }

  double energy(Utils::Vector3d const &vec1, Utils::Vector3d const &vec2) const {
    auto const cos_phi = calc_cosine(vec1, vec2, true);
    auto const phi = std::acos(cos_phi);
    return spline.evaluate(phi).first;
  }
};

struct ConservativeSplineDihedralBond {
  static constexpr int num = 3;
  ConservativeSpline1D spline;

  ConservativeSplineDihedralBond(double min, double max,
                                 std::vector<double> const &energy,
                                 std::vector<double> const &derivative)
      : spline(min, max, energy, derivative, true) {
    auto const two_pi = 2.0 * std::acos(-1.0);
    if (std::abs(min) > 1.0e-12 || std::abs(max - two_pi) > 1.0e-10) {
      throw std::invalid_argument("ConservativeSplineDihedral requires range 0..2*pi");
    }
  }

  double cutoff() const { return 0.0; }

  std::optional<std::tuple<Utils::Vector3d, Utils::Vector3d,
                           Utils::Vector3d, Utils::Vector3d>>
  forces(Utils::Vector3d const &v12, Utils::Vector3d const &v23,
         Utils::Vector3d const &v34) const {
    Utils::Vector3d v12Xv23, v23Xv34;
    double l_v12Xv23, l_v23Xv34;
    double phi, cos_phi;
    auto const angle_is_undefined = calc_dihedral_angle(
        v12, v23, v34, v12Xv23, l_v12Xv23, v23Xv34, l_v23Xv34,
        cos_phi, phi);
    if (angle_is_undefined) {
      return {};
    }

    auto const dU_dphi = spline.evaluate(phi).second;

    // Direct Cartesian gradient of ESPResSo's signed phi.  calc_dihedral_angle
    // returns normalized plane normals and their pre-normalization lengths.
    // This form is algebraically equivalent to ESPResSo's traditional
    // (geometry/sin(phi)) force expression away from planar configurations,
    // but it has no artificial 0/pi division and therefore remains regular
    // whenever the two defining planes themselves are well defined.
    auto const bnorm = v23.norm();
    auto const b2 = bnorm * bnorm;
    auto const grad1 = (-bnorm / l_v12Xv23) * v12Xv23;
    auto const grad4 = (bnorm / l_v23Xv34) * v23Xv34;
    auto const a = (v12 * v23) / b2;
    auto const c = (v34 * v23) / b2;
    auto const grad2 = -(1.0 + a) * grad1 + c * grad4;
    auto const grad3 = a * grad1 - (1.0 + c) * grad4;

    auto const force1 = -dU_dphi * grad1;
    auto const force2 = -dU_dphi * grad2;
    auto const force3 = -dU_dphi * grad3;
    auto const force4 = -dU_dphi * grad4;
    // ESPResSo bonded-dihedral return order is p2,p1,p3,p4.
    return std::make_tuple(force2, force1, force3, force4);
  }

  std::optional<double> energy(Utils::Vector3d const &v12,
                               Utils::Vector3d const &v23,
                               Utils::Vector3d const &v34) const {
    Utils::Vector3d v12Xv23, v23Xv34;
    double l_v12Xv23, l_v23Xv34;
    double phi, cos_phi;
    auto const angle_is_undefined = calc_dihedral_angle(
        v12, v23, v34, v12Xv23, l_v12Xv23, v23Xv34, l_v23Xv34,
        cos_phi, phi);
    if (angle_is_undefined) {
      return {};
    }
    return spline.evaluate(phi).first;
  }
};
