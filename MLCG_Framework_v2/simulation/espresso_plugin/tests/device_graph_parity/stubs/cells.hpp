#pragma once
#include "Particle.hpp"
#include "BoxGeometry.hpp"
#include <vector>
#include <random>
struct Distance {
  explicit Distance(Utils::Vector3d const &v) : vec21(v), dist2(v.norm2()) {}
  Utils::Vector3d vec21;
  double dist2;
};
struct ParticleRange {
  Particle *b, *e;
  Particle *begin() const { return b; }
  Particle *end() const { return e; }
};
template <typename T = void> struct VerletCriterion {};
struct CellStructure {
  std::vector<Particle> parts;  // locals (stored in shuffled order)
  std::vector<Particle> ghosts;
  BoxGeometry *box = nullptr;
  bool duplicate_some = true;
  ParticleRange local_particles() const {
    auto *p = const_cast<Particle *>(parts.data());
    return {p, p + parts.size()};
  }
  ParticleRange ghost_particles() const {
    auto *p = const_cast<Particle *>(ghosts.data());
    return {p, p + ghosts.size()};
  }
  template <class K, class V> void non_bonded_loop(K kernel, V const &) {
    std::mt19937 rng(7);
    for (std::size_t i = 0; i < parts.size(); ++i)
      for (std::size_t j = i + 1; j < parts.size(); ++j) {
        auto d = box->mi(parts[i].pos(), parts[j].pos());
        if (d.norm2() > 2.9 * 2.9) continue;  // emulate cutoff+skin list
        bool swap = rng() & 1u;
        if (!swap) kernel(parts[i], parts[j], Distance(d));
        else kernel(parts[j], parts[i], Distance(box->mi(parts[j].pos(), parts[i].pos())));
        if (duplicate_some && (rng() % 7u == 0u))  // periodic alias seen twice
          kernel(parts[i], parts[j], Distance(d));
      }
  }
};
