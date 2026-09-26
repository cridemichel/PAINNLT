#!/usr/bin/env python3
"""Diedro a flessione-torsione combinate (CBT) nel DihedralBond di ESPResSo.

Con mult < 0 il legame Dihedral(bend=K, mult=-n, phase=phi0) diventa

    V = K g(t1) g(t2) [1 - cos(n phi - phi0)]

con t1 l'angolo p1-p2-p3 e t2 l'angolo p2-p3-p4: l'idea del potenziale a
flessione-torsione combinate di Bulacu et al. (J. Chem. Theory Comput. 9, 3282
(2013), MARTINI), ma con g = 1 finche' sin t >= 0,3 e un raccordo C1 a zero
verso l'allineamento, g = x (2 - x) con x = sin^2 t / 0,09.  Il sin^3 di Bulacu
cambia il diedro ovunque: sul backbone del TEL26, con angoli di legame tipici
di 140-150 gradi, <sin^3 sin^3> vale 0,01-0,05 e la ricalibrazione di K
arrivava a 30-100 volte.  Il diedro a
coseno semplice ha forze ~ K / sin(t): quando tre siti si allineano divergono.
Nel TEL26 e' successo nei loop col residuo ML acceso (angolo 14T-15A-16G a
sin t = 0,009, forza 10^4).  Con il fattore g la forza va a zero in modo
liscio; il gradiente dei fattori angolari si scrive coi coseni, senza
singolarita'.  mult > 0 lascia il diedro di ESPResSo com'era.

Va applicato DOPO install_dihedral_undefined_zero.py.  Idempotente; fallisce se
non riconosce il sorgente.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SENTINEL = "MLCG CBT"
SENTINEL_V2 = "MLCG CBT (smorzamento angolare)"
HELPER_V1 = r"""/** MLCG CBT: fattore angolare sin^3(t) dell'angolo t fra -u e w (angolo al
 *  vertice comune dei legami u = p_b - p_a e w = p_c - p_b), la sua derivata
 *  rispetto a c = cos t e le derivate di c rispetto a u e w.
 */
inline void mlcg_cbt_angle(Utils::Vector3d const &u, Utils::Vector3d const &w,
                           double &g, double &dg_dc, Utils::Vector3d &dc_du,
                           Utils::Vector3d &dc_dw) {
  auto const lu = u.norm();
  auto const lw = w.norm();
  auto c = -(u * w) / (lu * lw);
  c = std::clamp(c, -1., 1.);
  auto const s2 = 1. - c * c;
  auto const s = std::sqrt(s2);
  g = s2 * s;
  dg_dc = -3. * c * s;
  dc_du = -w / (lu * lw) - (c / (lu * lu)) * u;
  dc_dw = -u / (lu * lw) - (c / (lw * lw)) * w;
}

"""

HELPER_ANCHOR = "/** Compute the four-body dihedral interaction force."
HELPER = r"""/** MLCG CBT (smorzamento angolare): fattore g(t) dell'angolo t fra -u e w
 *  (angolo al vertice comune dei legami u = p_b - p_a e w = p_c - p_b), la sua
 *  derivata rispetto a c = cos t e le derivate di c rispetto a u e w.
 *  Con x = sin^2 t / S0^2:  g = 1 per x >= 1,  g = x (2 - x) per x < 1.
 *  Il diedro resta quello di sempre finche' sin t >= S0 (t < 162,5 gradi per
 *  S0 = 0,3) e si spegne in modo C1 verso l'allineamento, dove la forza del
 *  coseno semplice divergerebbe come 1/sin t.
 */
inline constexpr double mlcg_cbt_s0 = 0.3;
inline void mlcg_cbt_angle(Utils::Vector3d const &u, Utils::Vector3d const &w,
                           double &g, double &dg_dc, Utils::Vector3d &dc_du,
                           Utils::Vector3d &dc_dw) {
  auto const lu = u.norm();
  auto const lw = w.norm();
  auto c = -(u * w) / (lu * lw);
  c = std::clamp(c, -1., 1.);
  auto const x = (1. - c * c) / (mlcg_cbt_s0 * mlcg_cbt_s0);
  if (x >= 1.) {
    g = 1.;
    dg_dc = 0.;
  } else {
    g = x * (2. - x);
    dg_dc = (2. / (mlcg_cbt_s0 * mlcg_cbt_s0)) * (1. - x) * (-2. * c);
  }
  dc_du = -w / (lu * lw) - (c / (lu * lu)) * u;
  dc_dw = -u / (lu * lw) - (c / (lw * lw)) * w;
}

"""

FORCE_MULT_OLD = "  auto const mult_ = static_cast<double>(mult);\n  auto const mphi = mult_ * phi - phase;\n  auto fac = -bend * mult_;"
FORCE_MULT_NEW = "  auto const mult_ = static_cast<double>(std::abs(mult)); /* MLCG CBT: mult < 0 */\n  auto const mphi = mult_ * phi - phase;\n  auto fac = -bend * mult_;"

FORCE_RET_OLD = "  return std::make_tuple(force2, force1, force3, -(force1 + force2 + force3));\n}"
FORCE_RET_NEW = r"""  /* MLCG CBT: V = bend g(t1) g(t2) [1 - cos(mphi)] se mult < 0 */
  if (mult < 0) {
    double g1, dg1, g2, dg2;
    Utils::Vector3d dc1_du, dc1_dw, dc2_du, dc2_dw;
    mlcg_cbt_angle(v12, v23, g1, dg1, dc1_du, dc1_dw);
    mlcg_cbt_angle(v23, v34, g2, dg2, dc2_du, dc2_dw);
    auto const e_phi = bend * (1. - std::cos(mphi));
    auto const a1 = e_phi * g2 * dg1;
    auto const a2 = e_phi * g1 * dg2;
    auto const force4 = -(force1 + force2 + force3);
    auto const s12 = g1 * g2;
    auto const cbt1 = s12 * force1 + a1 * dc1_du;
    auto const cbt2 = s12 * force2 - a1 * (dc1_du - dc1_dw) + a2 * dc2_du;
    auto const cbt3 = s12 * force3 - a1 * dc1_dw - a2 * (dc2_du - dc2_dw);
    auto const cbt4 = s12 * force4 - a2 * dc2_dw;
    return std::make_tuple(cbt2, cbt1, cbt3, cbt4);
  }

  return std::make_tuple(force2, force1, force3, -(force1 + force2 + force3));
}"""

ENERGY_OLD = "  auto const mphi = static_cast<double>(mult) * phi - phase;\n  return bend * (1. - std::cos(mphi));"
ENERGY_NEW = r"""  auto const mphi = static_cast<double>(std::abs(mult)) * phi - phase;
  if (mult < 0) { /* MLCG CBT */
    double g1, dg1, g2, dg2;
    Utils::Vector3d d1, d2, d3, d4;
    mlcg_cbt_angle(v12, v23, g1, dg1, d1, d2);
    mlcg_cbt_angle(v23, v34, g2, dg2, d3, d4);
    return bend * g1 * g2 * (1. - std::cos(mphi));
  }
  return bend * (1. - std::cos(mphi));"""

INCLUDE_OLD = "#include <cmath>\n"
INCLUDE_NEW = "#include <algorithm>\n#include <cmath>\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--espresso-root", required=True, type=Path)
    args = ap.parse_args()
    path = args.espresso_root / "src/core/bonded_interactions/dihedral.hpp"
    if not path.is_file():
        raise SystemExit(f"[ERROR] non trovo {path}")
    text = path.read_text()
    if SENTINEL_V2 in text:
        print(f"[SKIP] CBT (smorzamento angolare) gia' presente: {path}")
        return
    if SENTINEL in text:
        # prima versione (fattore sin^3 di Bulacu): si sostituisce solo il fattore
        if text.count(HELPER_V1) != 1:
            raise SystemExit(f"[ERROR] {path}: CBT presente ma non riconosciuta, patch non applicata")
        path.write_text(text.replace(HELPER_V1, HELPER, 1))
        print(f"[PASS] CBT aggiornata allo smorzamento angolare: {path}")
        return
    if "MLCG: diedro indefinito" not in text:
        raise SystemExit("[ERROR] applica prima install_dihedral_undefined_zero.py")
    for old in (HELPER_ANCHOR, FORCE_MULT_OLD, FORCE_RET_OLD, ENERGY_OLD, INCLUDE_OLD):
        if text.count(old) != 1:
            raise SystemExit(f"[ERROR] {path}: sorgente non riconosciuto ({old[:40]!r}...), patch non applicata")
    text = text.replace(INCLUDE_OLD, INCLUDE_NEW, 1)
    text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    text = text.replace(FORCE_MULT_OLD, FORCE_MULT_NEW, 1)
    text = text.replace(FORCE_RET_OLD, FORCE_RET_NEW, 1)
    text = text.replace(ENERGY_OLD, ENERGY_NEW, 1)
    path.write_text(text)
    print(f"[PASS] diedro CBT (mult < 0) installato: {path}")


if __name__ == "__main__":
    main()
