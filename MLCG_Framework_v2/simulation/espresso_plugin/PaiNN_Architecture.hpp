#pragma once
#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <string_view>
#include <stdexcept>

inline constexpr std::string_view PAINN_ARCHITECTURE_VARIANT = "painn_canonical_context_silu_v2";
inline constexpr std::string_view PAINN_ORDERED_GEOMETRY_VARIANT =
    "painn_ordered_geometry_tanh_v2";
inline constexpr std::string_view CGNET_ORDERED_GEOMETRY_VARIANT =
    "cgnet_ordered_geometry_tanh_v1";

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
            torch::nn::Linear(dim * 2, dim),
            torch::nn::SiLU(),
            torch::nn::Linear(dim, dim * 3)
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
    torch::nn::Sequential ordered_geometry_head{nullptr};
    int num_layers;
    int num_embeddings;
    double cutoff_radius; 
    int num_radial_basis; 
    double toxvaerd_alpha;
    int ordered_geometry_nodes;
    int ordered_geometry_feature_count;
    int ordered_geometry_head_layers;
    int ordered_geometry_head_width;
    bool ordered_geometry_head_only;
    torch::Tensor energy_scale;
    torch::Tensor ordered_geometry_energy_scale;
    torch::Tensor ordered_geometry_mean;
    torch::Tensor ordered_geometry_std;
    
    PaiNNModelImpl(
        int num_embeddings,
        int dim,
        int layers,
        int num_rbf = 20,
        double cutoff = 5.0,
        double t_alpha = 0.1,
        int ordered_nodes = 0,
        int ordered_head_layers = 5,
        int ordered_head_width = 160,
        double ordered_energy_scale_kj_mol = 0.0,
        bool ordered_head_only = false)
        : num_layers(layers),
          num_embeddings(num_embeddings),
          cutoff_radius(cutoff),
          num_radial_basis(num_rbf),
          toxvaerd_alpha(t_alpha),
          ordered_geometry_nodes(ordered_nodes),
          ordered_geometry_feature_count(0),
          ordered_geometry_head_layers(ordered_head_layers),
          ordered_geometry_head_width(ordered_head_width),
          ordered_geometry_head_only(ordered_head_only) {
        
        energy_scale = register_buffer("energy_scale", torch::ones({1}));
        if (!ordered_geometry_head_only) {
            embedding = register_module("embedding", torch::nn::Embedding(num_embeddings, dim));
            for (int i = 0; i < layers; ++i) {
                messages.push_back(register_module("message_" + std::to_string(i), PaiNNMessage(dim, num_rbf)));
                updates.push_back(register_module("update_" + std::to_string(i), PaiNNUpdate(dim)));
            }
            readout = register_module("readout", torch::nn::Sequential(
                torch::nn::Linear(dim, dim / 2), torch::nn::SiLU(), torch::nn::Linear(dim / 2, 1)
            ));
        }
        if (ordered_geometry_nodes > 0) {
            if (ordered_geometry_nodes < 4 || ordered_geometry_head_layers <= 0 ||
                ordered_geometry_head_width <= 0 ||
                !std::isfinite(ordered_energy_scale_kj_mol) ||
                ordered_energy_scale_kj_mol <= 0.0) {
                throw std::invalid_argument(
                    "Ordered geometry head requires at least four nodes, positive layer sizes, "
                    "and a positive finite energy scale");
            }
            ordered_geometry_feature_count =
                ordered_geometry_nodes * (ordered_geometry_nodes - 1) / 2 +
                (ordered_geometry_nodes - 2) + 2 * (ordered_geometry_nodes - 3);
            ordered_geometry_mean = register_buffer(
                "ordered_geometry_mean", torch::zeros({ordered_geometry_feature_count}));
            ordered_geometry_std = register_buffer(
                "ordered_geometry_std", torch::ones({ordered_geometry_feature_count}));
            ordered_geometry_energy_scale = register_buffer(
                "ordered_geometry_energy_scale",
                torch::full({1}, ordered_energy_scale_kj_mol, torch::kFloat32));

            ordered_geometry_head = register_module(
                "ordered_geometry_head", torch::nn::Sequential());
            const auto make_ordered_linear = [this](int input, int output) {
                auto linear = torch::nn::Linear(input, output);
                if (ordered_geometry_head_only) {
                    // Match cgnet.feature.utils.LinearLayer(weight_init="xavier"):
                    // overwrite only the weight and retain nn.Linear's default bias.
                    torch::NoGradGuard no_grad;
                    torch::nn::init::xavier_uniform_(linear->weight);
                }
                return linear;
            };
            ordered_geometry_head->push_back(make_ordered_linear(
                ordered_geometry_feature_count, ordered_geometry_head_width));
            ordered_geometry_head->push_back(torch::nn::Tanh());
            for (int layer = 1; layer < ordered_geometry_head_layers; ++layer) {
                ordered_geometry_head->push_back(make_ordered_linear(
                    ordered_geometry_head_width, ordered_geometry_head_width));
                ordered_geometry_head->push_back(torch::nn::Tanh());
            }
            ordered_geometry_head->push_back(make_ordered_linear(
                ordered_geometry_head_width, 1));
        } else if (ordered_geometry_head_only) {
            throw std::invalid_argument(
                "CGnet-exact ordered-head-only mode requires ordered geometry");
        }
    }

    bool has_ordered_geometry_head() const {
        return ordered_geometry_nodes > 0;
    }

    bool has_painn_branch() const {
        return !ordered_geometry_head_only;
    }

    void set_ordered_geometry_statistics(
        const torch::Tensor& feature_mean,
        const torch::Tensor& feature_std) {
        if (!has_ordered_geometry_head()) {
            throw std::runtime_error("Cannot set ordered geometry statistics on base PaiNN");
        }
        if (feature_mean.numel() != ordered_geometry_feature_count ||
            feature_std.numel() != ordered_geometry_feature_count) {
            throw std::runtime_error("Ordered geometry feature-statistics size mismatch");
        }
        ordered_geometry_mean.copy_(
            feature_mean.to(ordered_geometry_mean.device()).to(ordered_geometry_mean.dtype()));
        ordered_geometry_std.copy_(
            feature_std.to(ordered_geometry_std.device()).to(ordered_geometry_std.dtype()));
    }

    torch::Tensor ordered_geometry_features(
        const torch::Tensor& r_ij,
        const torch::Tensor& edge_index,
        const torch::Tensor& batch_indices) {
        if (!has_ordered_geometry_head()) {
            throw std::runtime_error("Ordered geometry features requested for base PaiNN");
        }
        const int64_t num_frames = batch_indices.max().cpu().item<int64_t>() + 1;
        const int64_t expected_nodes = num_frames * ordered_geometry_nodes;
        const int64_t expected_edges =
            num_frames * ordered_geometry_nodes * (ordered_geometry_nodes - 1);
        if (batch_indices.size(0) != expected_nodes || edge_index.size(1) != expected_edges) {
            throw std::runtime_error(
                "Ordered geometry head requires contiguous fixed-size frames and a complete directed graph");
        }

        auto row = edge_index.index({0});
        auto col = edge_index.index({1});
        auto edge_batches = batch_indices.index_select(0, row);
        auto local_row = row - edge_batches * ordered_geometry_nodes;
        auto local_col = col - edge_batches * ordered_geometry_nodes;
        auto dense = torch::zeros(
            {num_frames, ordered_geometry_nodes, ordered_geometry_nodes, 3}, r_ij.options());
        dense.index_put_({edge_batches, local_row, local_col}, r_ij);

        constexpr double eps = 1.0e-12;
        std::vector<torch::Tensor> features;
        features.reserve(ordered_geometry_feature_count);
        for (int i = 0; i < ordered_geometry_nodes; ++i) {
            for (int j = i + 1; j < ordered_geometry_nodes; ++j) {
                auto displacement = dense.index({torch::indexing::Slice(), i, j});
                features.push_back(torch::sqrt(
                    torch::sum(displacement * displacement, 1) + eps).unsqueeze(1));
            }
        }
        for (int i = 0; i < ordered_geometry_nodes - 2; ++i) {
            auto left = dense.index({torch::indexing::Slice(), i, i + 1});
            auto right = dense.index({torch::indexing::Slice(), i + 2, i + 1});
            auto denominator = torch::sqrt(
                torch::sum(left * left, 1) * torch::sum(right * right, 1) + eps);
            auto cosine = torch::sum(left * right, 1) / denominator;
            features.push_back(torch::acos(torch::clamp(cosine, -0.9999999, 0.9999999)).unsqueeze(1));
        }
        std::vector<torch::Tensor> dihedral_cosines;
        std::vector<torch::Tensor> dihedral_sines;
        for (int i = 0; i < ordered_geometry_nodes - 3; ++i) {
            auto b0 = dense.index({torch::indexing::Slice(), i + 1, i});
            auto b1 = dense.index({torch::indexing::Slice(), i + 2, i + 1});
            auto b2 = dense.index({torch::indexing::Slice(), i + 3, i + 2});
            auto normal_1 = torch::linalg_cross(b0, b1, 1);
            auto normal_2 = torch::linalg_cross(b1, b2, 1);
            auto normal_product = torch::sqrt(
                torch::sum(normal_1 * normal_1, 1) *
                torch::sum(normal_2 * normal_2, 1) + eps);
            auto cosine = torch::sum(normal_1 * normal_2, 1) / normal_product;
            auto b1_unit = b1 / torch::sqrt(torch::sum(b1 * b1, 1) + eps).unsqueeze(1);
            auto sine = torch::sum(
                torch::linalg_cross(normal_1, normal_2, 1) * b1_unit, 1) /
                normal_product;
            auto cosine_feature = torch::clamp(cosine, -1.0, 1.0).unsqueeze(1);
            auto sine_feature = torch::clamp(sine, -1.0, 1.0).unsqueeze(1);
            if (ordered_geometry_head_only) {
                dihedral_cosines.push_back(cosine_feature);
                dihedral_sines.push_back(sine_feature);
            } else {
                features.push_back(cosine_feature);
                features.push_back(sine_feature);
            }
        }
        if (ordered_geometry_head_only) {
            features.insert(features.end(), dihedral_cosines.begin(), dihedral_cosines.end());
            features.insert(features.end(), dihedral_sines.begin(), dihedral_sines.end());
        }
        return torch::cat(features, 1);
    }

    torch::Tensor ordered_geometry_energy(
        const torch::Tensor& r_ij,
        const torch::Tensor& edge_index,
        const torch::Tensor& batch_indices) {
        auto features = ordered_geometry_features(r_ij, edge_index, batch_indices);
        auto normalized = (features - ordered_geometry_mean) / ordered_geometry_std;
        auto raw = ordered_geometry_head->forward(normalized);
        auto reference = ordered_geometry_head->forward(torch::zeros_like(normalized));
        return (raw - reference) * ordered_geometry_energy_scale;
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
        return (raw_atom_energies - references.index_select(0, atomic_numbers)) * energy_scale;
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
        if (!has_painn_branch()) {
            return ordered_geometry_energy(r_ij, batch.edge_index, batch.batch_indices);
        }
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
        if (has_ordered_geometry_head()) {
            pred_energy = pred_energy + ordered_geometry_energy(
                r_ij, batch.edge_index, batch.batch_indices);
        }
        return pred_energy;
    }

    // --- FORWARD CON R_IJ ESPLICITO (PER ESPRESSO PBC) ---
    torch::Tensor forward_with_rij(torch::Tensor atomic_numbers, 
                                   torch::Tensor r_ij,
                                   torch::Tensor edge_index, 
                                   torch::Tensor batch_indices) {
        if (!has_painn_branch()) {
            return ordered_geometry_energy(r_ij, edge_index, batch_indices).squeeze(-1);
        }
        
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
        if (has_ordered_geometry_head()) {
            pred_energy = pred_energy + ordered_geometry_energy(
                r_ij, edge_index, batch_indices);
        }
        
        return pred_energy.squeeze(-1); 
    }
    
    // --- FORWARD PER ATOMO (PER MPI GHOST PARTICLE FILTERING) ---
    torch::Tensor forward_atom_energies(torch::Tensor atomic_numbers, 
                                        torch::Tensor r_ij,
                                        torch::Tensor edge_index) {
        if (!has_painn_branch()) {
            auto batch_indices = torch::zeros(
                {atomic_numbers.size(0)}, atomic_numbers.options());
            auto head_energy = ordered_geometry_energy(r_ij, edge_index, batch_indices);
            return torch::ones(
                {atomic_numbers.size(0), 1}, head_energy.options()) *
                head_energy.index({0, 0}) / static_cast<double>(ordered_geometry_nodes);
        }
        
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

        auto atom_energies = apply_isolated_species_gauge(
            atomic_numbers, readout->forward(s));
        if (has_ordered_geometry_head()) {
            auto batch_indices = torch::zeros(
                {atomic_numbers.size(0)}, atomic_numbers.options());
            auto head_energy = ordered_geometry_energy(r_ij, edge_index, batch_indices);
            atom_energies = atom_energies +
                head_energy.index({0, 0}) / static_cast<double>(ordered_geometry_nodes);
        }
        return atom_energies;
    }
};
TORCH_MODULE(PaiNNModel);
