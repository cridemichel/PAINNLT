#include "../training/PaiNN_Architecture.hpp"

#include <cmath>
#include <iostream>

int main() {
    torch::manual_seed(1234);
    PaiNNModel model(3, 16, 2, 12, 1.0, 0.1);
    model->eval();

    auto species = torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kInt64));
    auto empty_r = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto empty_edges = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64));
    auto zero_edge = model->forward_atom_energies(species, empty_r, empty_edges).squeeze(-1);

    if (zero_edge.sizes() != torch::IntArrayRef({2})) {
        std::cerr << "unexpected zero-edge energy shape: " << zero_edge.sizes() << "\n";
        return 1;
    }
    if (!torch::isfinite(zero_edge).all().item<bool>()) {
        std::cerr << "non-finite zero-edge energies\n";
        return 2;
    }
    const double zero_edge_max = zero_edge.abs().max().item<double>();
    if (zero_edge_max > 1.0e-6) {
        std::cerr << "isolated-species gauge did not zero the baseline: "
                  << zero_edge << "\n";
        return 3;
    }

    auto r_ij = torch::tensor(
        {{0.35f, 0.0f, 0.0f}, {-0.35f, 0.0f, 0.0f}},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    auto edges = torch::tensor(
        {{0, 1}, {1, 0}}, torch::TensorOptions().dtype(torch::kInt64));
    auto shifted = model->forward_atom_energies(species, r_ij, edges).squeeze(-1);
    auto references = model->isolated_species_reference_table(species)
                          .index_select(0, species)
                          .squeeze(-1);
    auto shifted_total = shifted.to(torch::kFloat64).sum();
    auto reconstructed_raw_total = (shifted + references).to(torch::kFloat64).sum();

    auto shifted_grad = torch::autograd::grad(
        {shifted_total}, {r_ij}, {torch::ones_like(shifted_total)}, true, false)[0];
    auto raw_grad = torch::autograd::grad(
        {reconstructed_raw_total}, {r_ij}, {torch::ones_like(reconstructed_raw_total)}, false, false)[0];
    const double gradient_difference = (shifted_grad - raw_grad).abs().max().item<double>();
    if (gradient_difference > 1.0e-6) {
        std::cerr << "energy gauge changed coordinate gradients: "
                  << gradient_difference << "\n";
        return 4;
    }

    std::cout << "zero_edge_max_abs=" << zero_edge_max << "\n";
    std::cout << "raw_reference_max_abs=" << references.abs().max().item<double>() << "\n";
    std::cout << "gauge_gradient_difference=" << gradient_difference << "\n";
    return 0;
}
