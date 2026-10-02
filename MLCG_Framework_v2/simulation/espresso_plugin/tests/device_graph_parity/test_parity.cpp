#include "PaiNN_ML_Potential.hpp"
#include "system/System.hpp"
#include <cstdlib>
#include <iostream>
#include <random>

int main(int argc, char **argv) {
  const std::string precision = argc > 1 ? argv[1] : "float32";
  const int nspec = 8, nmol = 40, per_mol = 8;
  const double L = 6.0, rc = 1.26;
  torch::manual_seed(3);
  {
    PaiNNModel m(nspec, 32, 3, 16, rc, 0.3);
    torch::save(m, "/tmp/claude_painn_test.pt");
  }
  auto &box = *System::get_system().box_geo;
  box.L[0] = L; box.L[1] = L * 1.1; box.L[2] = L * 0.95;
  CellStructure cs; cs.box = &box;
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(-0.15, 1.15);  // some outside box
  std::vector<int> ids(nmol * per_mol);
  for (std::size_t k = 0; k < ids.size(); ++k) ids[k] = int(k) * 3 + 5;
  std::shuffle(ids.begin(), ids.end(), rng);
  for (int m = 0; m < nmol; ++m) {
    Utils::Vector3d c; for (int a = 0; a < 3; ++a) c[a] = u(rng) * box.L[a];
    for (int s = 0; s < per_mol; ++s) {
      Particle p; p.m_id = ids[m * per_mol + s]; p.m_type = (m + s) % nspec; p.m_mol = m;
      for (int a = 0; a < 3; ++a) p.r[a] = c[a] + 0.35 * (u(rng) - 0.5);
      cs.parts.push_back(p);
    }
  }
  // a non-ML particle (type >= nspec) must be ignored by both paths
  Particle marker; marker.m_id = 99999; marker.m_type = nspec + 1; marker.m_mol = 777;
  cs.parts.push_back(marker);
  // periodic-alias ghosts of locals (allowed)
  for (int k = 0; k < 10; ++k) cs.ghosts.push_back(cs.parts[k]);

  auto run = [&](const char *mode, double &energy, std::string *prof, const char *stat = "0") {
    setenv("MLCG_PAINN_GRAPH", mode, 1);
    setenv("MLCG_PAINN_CUDA_GRAPH", stat, 1);
    PaiNN_ML_Potential pot("/tmp/claude_painn_test.pt", nspec, 32, 3, 16, rc, 0.3,
                           0, 0, 0, 0.0, false, 1, false, "cpu", precision);
    pot.configure_profiling(true, 0);
    for (auto &p : cs.parts) p.f = Utils::Vector3d{};
    VerletCriterion<> vc;
    pot.calculate_forces(cs, vc);
    // second call with identical state exercises the cached path
    for (auto &p : cs.parts) p.f = Utils::Vector3d{};
    pot.calculate_forces(cs, vc);
    energy = pot.get_last_energy();
    if (prof) *prof = pot.get_profile_json();
    std::vector<double> f;
    for (auto &p : cs.parts) for (int a = 0; a < 3; ++a) f.push_back(p.f[a]);
    return f;
  };
  double el, ed, es; std::string pj, ps;
  auto fl = run("legacy", el, nullptr);
  auto fd = run("device", ed, &pj);
  auto fs = run("device", es, &ps, "1");
  {
    double m = 0, mf = 0;
    for (std::size_t k = 0; k < fl.size(); ++k) { m = std::max(m, std::abs(fl[k] - fs[k])); mf = std::max(mf, std::abs(fl[k])); }
    std::cout.precision(12);
    std::cout << "static(padded) vs legacy: |dE|=" << std::abs(el - es) << " max|dF|=" << m << " rel=" << m / mf << "\n";
    std::cout << " static profile: " << ps.substr(ps.find("\"static_graph\""), 80) << "\n";
  }
  setenv("MLCG_PAINN_CUDA_GRAPH", "0", 1);
  double maxdf = 0, maxf = 0, sumf[3] = {0, 0, 0};
  for (std::size_t k = 0; k < fl.size(); ++k) {
    maxdf = std::max(maxdf, std::abs(fl[k] - fd[k]));
    maxf = std::max(maxf, std::abs(fl[k]));
    sumf[k % 3] += fd[k];
  }
  std::cout.precision(12);
  std::cout << "precision=" << precision << " E_legacy=" << el << " E_device=" << ed
            << " |dE|=" << std::abs(el - ed) << "\n max|F|=" << maxf << " max|dF|=" << maxdf
            << " rel=" << maxdf / maxf << "\n net force device=" << sumf[0] << "," << sumf[1]
            << "," << sumf[2] << "\n marker force=" << fd[fd.size() - 3] << "\n";
  std::cout << pj.substr(0, 400) << "\n";
  // finite-difference check of device forces on one particle
  setenv("MLCG_PAINN_GRAPH", "device", 1);
  PaiNN_ML_Potential pot("/tmp/claude_painn_test.pt", nspec, 32, 3, 16, rc, 0.3,
                         0, 0, 0, 0.0, false, 1, false, "cpu", precision);
  VerletCriterion<> vc;
  const int k = 3; const double h = precision == "float64" ? 1e-5 : 2e-3;
  double fd_err = 0;
  for (int a = 0; a < 3; ++a) {
    double x0 = cs.parts[k].r[a];
    cs.parts[k].r[a] = x0 + h; pot.calculate_forces(cs, vc); double ep = pot.get_last_energy();
    cs.parts[k].r[a] = x0 - h; pot.calculate_forces(cs, vc); double em = pot.get_last_energy();
    cs.parts[k].r[a] = x0;
    double fnum = -(ep - em) / (2 * h);
    fd_err = std::max(fd_err, std::abs(fnum - fd[3 * k + a]));
    std::cout << " FD axis " << a << ": analytic=" << fd[3 * k + a] << " numeric=" << fnum << "\n";
  }
  return 0;
}
