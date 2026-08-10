#pragma once
#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <string_view>

inline constexpr std::string_view PAINN_ARCHITECTURE_VARIANT = "painn_canonical_context_silu_v2";

// ============================================================================
// 1. BLOCCO MESSAGGI (Message Passing)
// ============================================================================
struct PaiNNMessageImpl : torch::nn::Module {
    torch::nn::Sequential scalar_mlp{nullptr};
    torch::nn::Linear filter_mlp{nullptr};
    PaiNNMessageImpl(int dim, int num_rbf) {
        // Canonical PaiNN interatomic context network:
        // D -> D with SiLU -> 3D, then elementwise multiplication by the
        // radial filter.  The previous single Linear(D, 3D) removed the
        // nonlinear context transform and materially reduced data efficiency.
        scalar_mlp = register_module("scalar_mlp", torch::nn::Sequential(
            torch::nn::Linear(dim, dim),
            torch::nn::SiLU(),
            torch::nn::Linear(dim, dim * 3)
        ));
        filter_mlp = register_module("filter_mlp", torch::nn::Linear(
            torch::nn::LinearOptions(num_rbf, dim * 3).bias(false)));
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor s, torch::Tensor v, torch::Tensor edge_index, torch::Tensor rbf, torch::Tensor r_ij_norm) {
        auto row = edge_index[0], col = edge_index[1]; 
        auto w = filter_mlp->forward(rbf); 
        auto interaction = scalar_mlp->forward(s.index({row})) * w;      
        auto chunks = interaction.chunk(3, 1);
        auto delta_v_edges = v.index({row}) * chunks[1].unsqueeze(1) + chunks[2].unsqueeze(1) * r_ij_norm.unsqueeze(2);
        auto delta_s = torch::zeros_like(s);
        auto delta_v = torch::zeros_like(v);
        delta_s.index_add_(0, col, chunks[0]);
        delta_v.index_add_(0, col, delta_v_edges);
        return {delta_s, delta_v};
    }
};
TORCH_MODULE(PaiNNMessage);

// ============================================================================
// 2. BLOCCO AGGIORNAMENTO (Update Block)
// ============================================================================
struct PaiNNUpdateImpl : torch::nn::Module {
    torch::nn::Linear linear_v{nullptr}, linear_u{nullptr};
    torch::nn::Sequential scalar_mlp{nullptr};
    double epsilon;
    PaiNNUpdateImpl(int dim, double eps = 1.0e-8) : epsilon(eps) {
        linear_v = register_module("linear_v", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
        linear_u = register_module("linear_u", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
        scalar_mlp = register_module("scalar_mlp", torch::nn::Sequential(
            torch::nn::Linear(dim * 2, dim), torch::nn::SiLU(), torch::nn::Linear(dim, dim * 3)
        ));
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor s, torch::Tensor v) {
        auto v_v = linear_v->forward(v);
        auto v_u = linear_u->forward(v);
        // Match the canonical PaiNN stabilized vector norm.  The epsilon is
        // important when vector features are exactly zero (e.g. isolated
        // species or the first interaction block) and keeps second-order
        // force training numerically smooth.
        auto v_v_norm = torch::sqrt(torch::sum(v_v * v_v, 1) + epsilon);
        auto s_out = scalar_mlp->forward(torch::cat({s, v_v_norm}, 1));
        auto chunks = s_out.chunk(3, 1);
        auto delta_s = chunks[0] + (v_v * v_u).sum(1) * chunks[1]; 
        return {s + delta_s, v + v_u * chunks[2].unsqueeze(1)};
    }
};
TORCH_MODULE(PaiNNUpdate);

// ============================================================================
// 3. MODELLO COMPLETO (PaiNN)
// ============================================================================
struct PaiNNModelImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    std::vector<PaiNNMessage> messages;
    std::vector<PaiNNUpdate> updates;
    torch::nn::Sequential readout{nullptr};
    int num_layers;
    int num_embeddings;
    double cutoff_radius; 
    int num_radial_basis; 
    double toxvaerd_alpha;
    
    PaiNNModelImpl(int num_embeddings, int dim, int layers, int num_rbf = 20, double cutoff = 5.0, double t_alpha = 0.1) 
        : num_layers(layers), num_embeddings(num_embeddings), cutoff_radius(cutoff), num_radial_basis(num_rbf), toxvaerd_alpha(t_alpha) {
        
        embedding = register_module("embedding", torch::nn::Embedding(num_embeddings, dim));
        for (int i = 0; i < layers; ++i) {
            messages.push_back(register_module("message_" + std::to_string(i), PaiNNMessage(dim, num_rbf)));
            updates.push_back(register_module("update_" + std::to_string(i), PaiNNUpdate(dim)));
        }
        readout = register_module("readout", torch::nn::Sequential(
            torch::nn::Linear(dim, dim / 2), torch::nn::SiLU(), torch::nn::Linear(dim / 2, 1)
        ));
    }


    // Force-only training leaves the additive energy gauge unconstrained.
    // We fix it by assigning zero energy to every isolated site type.  The
    // subtraction depends on model parameters and species, but not on
    // coordinates, so forces and force-training gradients are unchanged.
    torch::Tensor isolated_species_reference_table(torch::Tensor atomic_numbers) {
        auto species = torch::arange(num_embeddings, atomic_numbers.options());
        torch::Tensor s = embedding->forward(species);
        torch::Tensor v = torch::zeros({s.size(0), 3, s.size(1)}, s.options());
        for (int i = 0; i < num_layers; ++i) {
            std::tie(s, v) = updates[i]->forward(s, v);
        }
        return readout->forward(s);
    }

    torch::Tensor apply_isolated_species_gauge(
        torch::Tensor atomic_numbers, torch::Tensor raw_atom_energies) {
        auto references = isolated_species_reference_table(atomic_numbers);
        return raw_atom_energies - references.index_select(0, atomic_numbers);
    }

    // Espansione RBF con Toxvaerd Cutoff (C3 smooth energy)
    torch::Tensor expansion_rbf(torch::Tensor d_ij) {
        double r_c = cutoff_radius; 
        
        auto x = (r_c - d_ij) / r_c;
        auto x_n = torch::pow(x, 4);
        auto tox_cutoff = x_n / (x_n + std::pow(toxvaerd_alpha, 4));
        tox_cutoff = torch::where(d_ij > r_c, torch::zeros_like(tox_cutoff), tox_cutoff);

        auto centers = torch::linspace(0.0, r_c, num_radial_basis, d_ij.options());
        double sigma = r_c / num_radial_basis;
        auto rbf = torch::exp(-torch::pow(d_ij.unsqueeze(1) - centers, 2) / torch::pow(torch::full_like(centers, sigma), 2));
        return rbf * tox_cutoff.unsqueeze(1);
    }
    
    // --- FORWARD COMPATIBILE CON PAINN.CPP (TRAINING) ---
    template <typename BatchType>
    torch::Tensor forward(BatchType& batch) {
        auto row = batch.edge_index[0], col = batch.edge_index[1]; 
        auto r_ij = batch.coordinates.index({row}) - batch.coordinates.index({col});
        auto d_ij = torch::sqrt(torch::sum(r_ij * r_ij, 1) + 1e-8);

        torch::Tensor s = embedding->forward(batch.atomic_numbers);
        torch::Tensor v = torch::zeros({s.size(0), 3, s.size(1)}, s.options());

        auto r_ij_norm = r_ij / d_ij.unsqueeze(1);
        auto rbf = expansion_rbf(d_ij);

        for (int i = 0; i < num_layers; ++i) {
            auto msg_out = messages[i]->forward(s, v, batch.edge_index, rbf, r_ij_norm);
            s = s + msg_out.first; 
            v = v + msg_out.second;
            
            std::tie(s, v) = updates[i]->forward(s, v);
        }

        torch::Tensor atom_energies = apply_isolated_species_gauge(
            batch.atomic_numbers, readout->forward(s));
        torch::Tensor pred_energy = torch::zeros({batch.energy_true.size(0), 1}, s.options());
        pred_energy.index_add_(0, batch.batch_indices, atom_energies);
        return pred_energy;
    }

    // --- FORWARD CON R_IJ ESPLICITO (PER ESPRESSO PBC) ---
    torch::Tensor forward_with_rij(torch::Tensor atomic_numbers, 
                                   torch::Tensor r_ij,
                                   torch::Tensor edge_index, 
                                   torch::Tensor batch_indices) {
        
        auto d_ij = torch::sqrt(torch::sum(r_ij * r_ij, 1) + 1e-8);
        
        torch::Tensor s = embedding->forward(atomic_numbers);
        torch::Tensor v = torch::zeros({s.size(0), 3, s.size(1)}, s.options());

        auto r_ij_norm = r_ij / d_ij.unsqueeze(1);
        auto rbf = expansion_rbf(d_ij);

        for (int i = 0; i < num_layers; ++i) {
            auto msg_out = messages[i]->forward(s, v, edge_index, rbf, r_ij_norm);
            s = s + msg_out.first; 
            v = v + msg_out.second;
            
            std::tie(s, v) = updates[i]->forward(s, v);
        }

        torch::Tensor atom_energies = apply_isolated_species_gauge(
            atomic_numbers, readout->forward(s));
        
        int64_t num_molecules = batch_indices.max().cpu().item<int64_t>() + 1;
        torch::Tensor pred_energy = torch::zeros({num_molecules, 1}, s.options());
        pred_energy.index_add_(0, batch_indices, atom_energies);
        
        return pred_energy.squeeze(-1); 
    }
    
    // --- FORWARD PER ATOMO (PER MPI GHOST PARTICLE FILTERING) ---
    torch::Tensor forward_atom_energies(torch::Tensor atomic_numbers, 
                                        torch::Tensor r_ij,
                                        torch::Tensor edge_index) {
        
        auto d_ij = torch::sqrt(torch::sum(r_ij * r_ij, 1) + 1e-8);
        
        torch::Tensor s = embedding->forward(atomic_numbers);
        torch::Tensor v = torch::zeros({s.size(0), 3, s.size(1)}, s.options());

        auto r_ij_norm = r_ij / d_ij.unsqueeze(1);
        auto rbf = expansion_rbf(d_ij);

        for (int i = 0; i < num_layers; ++i) {
            auto msg_out = messages[i]->forward(s, v, edge_index, rbf, r_ij_norm);
            s = s + msg_out.first; 
            v = v + msg_out.second;
            
            std::tie(s, v) = updates[i]->forward(s, v);
        }

        return apply_isolated_species_gauge(
            atomic_numbers, readout->forward(s));
    }
};
TORCH_MODULE(PaiNNModel);
