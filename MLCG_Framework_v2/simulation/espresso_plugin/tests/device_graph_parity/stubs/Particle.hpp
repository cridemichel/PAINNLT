#pragma once
#include <cmath>
namespace Utils {
struct Vector3d {
  double v[3]{0, 0, 0};
  double &operator[](unsigned i) { return v[i]; }
  double operator[](unsigned i) const { return v[i]; }
  double norm2() const { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }
};
} // namespace Utils
struct Particle {
  int m_id = 0, m_type = 0, m_mol = 0;
  Utils::Vector3d r, f;
  int const &id() const { return m_id; }
  int const &type() const { return m_type; }
  int const &mol_id() const { return m_mol; }
  Utils::Vector3d const &pos() const { return r; }
  Utils::Vector3d &force() { return f; }
};
