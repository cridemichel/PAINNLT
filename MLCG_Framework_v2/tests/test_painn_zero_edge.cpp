#include "../training/PaiNN_Architecture.hpp"

#include <cmath>
#include <iostream>

int main() {
    torch::manual_seed(1234);
    PaiNNModel model(3, 16, 2, 12, 1.0, 0.1);
    model->eval();

    auto species = torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kInt64));
    auto r_ij = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto edges = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64));
    auto energies = model->forward_atom_energies(species, r_ij, edges).squeeze(-1);

    if (energies.sizes() != torch::IntArrayRef({2})) {
        std::cerr << "unexpected zero-edge energy shape: " << energies.sizes() << "\n";
        return 1;
    }
    if (!torch::isfinite(energies).all().item<bool>()) {
        std::cerr << "non-finite zero-edge energies\n";
        return 2;
    }
    std::cout << "zero_edge_energy=" << energies.sum().item<double>() << "\n";
    return 0;
}
