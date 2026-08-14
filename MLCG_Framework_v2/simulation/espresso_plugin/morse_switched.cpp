/* MLCG extension of ESPResSo's non-bonded Morse parameters. */
#include <config/config.hpp>

#ifdef ESPRESSO_MORSE

#include "nonbonded_interaction_data.hpp"

#include <cmath>
#include <stdexcept>

Morse_Parameters::Morse_Parameters(double eps, double alpha, double rmin,
                                   double cutoff, double switch_start)
    : eps{eps}, alpha{alpha}, rmin{rmin}, cut{cutoff},
      switch_start{switch_start} {
  if (eps < 0.) {
    throw std::domain_error("Morse parameter 'eps' has to be >= 0");
  }
  if (cutoff < 0.) {
    throw std::domain_error("Morse parameter 'cutoff' has to be >= 0");
  }

  if (switch_start >= 0.0) {
    if (alpha <= 0.0) {
      throw std::domain_error(
          "Switched Morse parameter 'alpha' has to be > 0");
    }
    if (!(rmin < switch_start && switch_start < cutoff)) {
      throw std::domain_error(
          "Switched Morse requires rmin < switch_start < cutoff");
    }
    // Not used in switched mode, but keep the member finite and deterministic.
    rest = 0.0;
  } else {
    // Preserve ESPResSo's stock Morse energy shift when switching is disabled.
    auto const add1 = std::exp(-2.0 * alpha * (cut - rmin));
    auto const add2 = 2.0 * std::exp(-alpha * (cut - rmin));
    rest = eps * (add1 - add2);
  }
}

#endif // ESPRESSO_MORSE
