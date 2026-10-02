#pragma once
#include "Particle.hpp"
struct BoxGeometry {
  Utils::Vector3d L;
  bool per[3]{true, true, true};
  Utils::Vector3d const &length() const { return L; }
  constexpr bool periodic(unsigned c) const { return per[c]; }
  Utils::Vector3d mi(Utils::Vector3d const &a, Utils::Vector3d const &b) const {
    Utils::Vector3d d;
    for (unsigned c = 0; c < 3; ++c) {
      double dx = a[c] - b[c];
      if (per[c] && std::abs(dx) > 0.5 * L[c]) dx -= std::round(dx / L[c]) * L[c];
      d[c] = dx;
    }
    return d;
  }
};
