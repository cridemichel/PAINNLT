// Test delle teste termodinamiche di PaiNN (PaiNN_Architecture.hpp).
//
//   U(x;T) = sum_i eps [ a_i + (T - T0)/T0 * b_i ]   (a = readout, b = readout_dT)
//
// Controlla, in float64 su CPU:
//   1. un modello con teste caricato da un file senza teste (promozione) da' la
//      stessa energia del modello originale a ogni T;
//   2. con readout_dT non nulla: U(T) = A + (T-T0)/T0 B (per molecola e per
//      sito), dU/dT = B/T0, forze autograd = differenze finite;
//   3. salvataggio e ricaricamento (la T corrente resta nel file, il plugin la
//      reimposta sempre);
//   4. T0 diversa fra file e config rifiutata; set_temperature su un modello
//      senza teste rifiutata.
//
// Compilazione (esempio, LibTorch da pip):
//   TP=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
//   g++ -O2 -std=c++20 tests/test_painn_thermo_heads.cpp -Itraining -I$TP/include \
//       -I$TP/include/torch/csrc/api/include -L$TP/lib -Wl,-rpath,$TP/lib \
//       -ltorch_cpu -lc10 -o test_painn_thermo_heads && ./test_painn_thermo_heads
#include "PaiNN_Architecture.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static int fails = 0;
static void check(bool ok, const std::string& what, double val) {
    std::cout << (ok ? "[OK]   " : "[FAIL] ") << what << "  " << val << "\n";
    if (!ok) ++fails;
}

int main() {
    torch::manual_seed(7);
    const int64_t N = 12;
    auto types = torch::randint(8, {N}, torch::kInt64);
    auto pos = torch::rand({N, 3}, torch::kFloat64) * 1.2;
    std::vector<int64_t> r, c;
    for (int64_t i = 0; i < N; ++i)
        for (int64_t j = 0; j < N; ++j)
            if (i != j && (i / 2) != (j / 2)) { r.push_back(i); c.push_back(j); }
    auto edge = torch::stack({torch::tensor(r), torch::tensor(c)});
    auto rij = pos.index_select(0, edge[0]) - pos.index_select(0, edge[1]);
    auto batch = torch::zeros({N}, torch::kInt64);

    PaiNNModel base(8, 32, 2, 16, 1.26, 0.1);
    {
        torch::NoGradGuard g;
        for (auto& p : base->parameters()) p.add_(torch::randn_like(p) * 0.05);
        base->energy_scale.fill_(2.49);
    }
    torch::save(base, "thermo_test_base.pt");   // float32, come in produzione
    base->to(torch::kFloat64);
    const double eref = base->forward_with_rij(types, rij, edge, batch).item<double>();

    PaiNNModel mt(8, 32, 2, 16, 1.26, 0.1, 0, 5, 160, 0.0, false, 1, false, true, 300.0);
    const bool promoted = load_painn_weights(mt, "thermo_test_base.pt", 32);
    check(promoted, "promozione da un modello senza teste", promoted);
    mt->to(torch::kFloat64);
    for (double T : {250.0, 300.0, 400.0}) {
        mt->set_temperature(T);
        const double e = mt->forward_with_rij(types, rij, edge, batch).item<double>();
        check(std::abs(e - eref) < 1e-12 * std::max(1.0, std::abs(eref)),
              "testa nulla == modello originale a T=" + std::to_string(int(T)) + ", diff", e - eref);
    }

    {
        torch::NoGradGuard g;
        for (auto& p : mt->readout_dT->parameters()) p.add_(torch::randn_like(p) * 0.2);
    }
    auto parts = mt->forward_thermo_parts(types, rij, edge, batch);
    const double A = parts[0][0].item<double>(), B = parts[0][1].item<double>();
    double maxlin = 0.0;
    for (double T : {280.0, 330.0, 400.0}) {
        mt->set_temperature(T);
        const double u = mt->forward_with_rij(types, rij, edge, batch).item<double>();
        const double ua = mt->forward_atom_energies(types, rij, edge).sum().item<double>();
        maxlin = std::max({maxlin, std::abs(u - (A + (T - 300.0) / 300.0 * B)), std::abs(ua - u)});
    }
    check(maxlin < 1e-10, "U(T) = A + (T-T0)/T0 B, per molecola e per sito, max err", maxlin);
    check(std::abs(B) > 1e-6, "testa entropica non nulla, B", B);
    mt->set_temperature(350.0);
    const double u1 = mt->forward_with_rij(types, rij, edge, batch).item<double>();
    mt->set_temperature(350.5);
    const double u2 = mt->forward_with_rij(types, rij, edge, batch).item<double>();
    check(std::abs((u2 - u1) / 0.5 - B / 300.0) < 1e-9, "dU/dT = B/T0, diff", (u2 - u1) / 0.5 - B / 300.0);

    mt->set_temperature(370.0);
    auto rg = rij.clone().set_requires_grad(true);
    auto U = mt->forward_with_rij(types, rg, edge, batch).sum();
    auto grad = torch::autograd::grad({U}, {rg})[0];
    auto dir = torch::randn_like(rij);
    dir = dir / dir.norm();
    const double h = 1e-5;
    const double up = mt->forward_with_rij(types, rij + h * dir, edge, batch).item<double>();
    const double dn = mt->forward_with_rij(types, rij - h * dir, edge, batch).item<double>();
    const double fd = (up - dn) / (2 * h), an = (grad * dir).sum().item<double>();
    check(std::abs(fd - an) < 1e-6 * std::max(1.0, std::abs(an)), "forze: autograd contro differenze finite, rel",
          std::abs(fd - an) / std::max(1e-12, std::abs(an)));

    mt->to(torch::kFloat32);
    torch::save(mt, "thermo_test_heads.pt");
    mt->to(torch::kFloat64);
    const double uref = mt->forward_with_rij(types, rij, edge, batch).item<double>();
    PaiNNModel mt2(8, 32, 2, 16, 1.26, 0.1, 0, 5, 160, 0.0, false, 1, false, true, 300.0);
    const bool p2 = load_painn_weights(mt2, "thermo_test_heads.pt", 32);
    mt2->to(torch::kFloat64);
    const double u3 = mt2->forward_with_rij(types, rij, edge, batch).item<double>();
    check(!p2 && std::abs(u3 - uref) < 1e-12, "salva/carica (T del file 370 K), diff", u3 - uref);

    PaiNNModel mt3(8, 32, 2, 16, 1.26, 0.1, 0, 5, 160, 0.0, false, 1, false, true, 310.0);
    bool caught = false;
    try { load_painn_weights(mt3, "thermo_test_heads.pt", 32); } catch (const std::exception&) { caught = true; }
    check(caught, "T0 diversa fra file e config rifiutata", caught);
    caught = false;
    try { base->set_temperature(300.0); } catch (const std::logic_error&) { caught = true; }
    check(caught, "set_temperature su modello senza teste rifiutata", caught);

    std::cout << (fails ? "FALLITO\n" : "TUTTI I TEST PASSATI\n");
    return fails ? 1 : 0;
}
