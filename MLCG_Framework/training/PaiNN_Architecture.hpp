#pragma once
#include <torch/torch.h>
#include <cmath>
#include <vector>

// ============================================================================
// 1. BLOCCO MESSAGGI (Message Passing)
// ============================================================================
struct PaiNNMessageImpl : torch::nn::Module {
    torch::nn::Linear scalar_mlp{nullptr}, filter_mlp{nullptr};
    PaiNNMessageImpl(int dim, int num_rbf) { 
        scalar_mlp = register_module("scalar_mlp", torch::nn::Linear(dim, dim * 3));
        // AGGIUNTA: Sostituito 20 con num_rbf
        filter_mlp = register_module("filter_mlp", torch::nn::Linear(num_rbf, dim * 3)); 
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
    PaiNNUpdateImpl(int dim) {
        linear_v = register_module("linear_v", torch::nn::Linear(dim, dim));
        linear_u = register_module("linear_u", torch::nn::Linear(dim, dim));
        scalar_mlp = register_module("scalar_mlp", torch::nn::Sequential(
            torch::nn::Linear(dim * 2, dim), torch::nn::SiLU(), torch::nn::Linear(dim, dim * 3)
        ));
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor s, torch::Tensor v) {
        auto v_v = linear_v->forward(v); 
        auto v_u = linear_u->forward(v); 
        auto s_out = scalar_mlp->forward(torch::cat({s, (v_v * v_v).sum(1)}, 1)); 
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
    double cutoff_radius; 
    int num_radial_basis; 
    PaiNNModelImpl(int num_embeddings, int dim, int layers, int num_rbf = 20, double cutoff = 5.0) 
        : num_layers(layers), cutoff_radius(cutoff), num_radial_basis(num_rbf) {
        
        embedding = register_module("embedding", torch::nn::Embedding(num_embeddings, dim));
        for (int i = 0; i < layers; ++i) {
            messages.push_back(register_module("message_" + std::to_string(i), PaiNNMessage(dim, num_rbf)));
            updates.push_back(register_module("update_" + std::to_string(i), PaiNNUpdate(dim)));
        }
        readout = register_module("readout", torch::nn::Sequential(
            torch::nn::Linear(dim, dim / 2), torch::nn::SiLU(), torch::nn::Linear(dim / 2, 1)
        ));
    }

    // Espansione RBF con Cosine Cutoff integrato per stabilità dinamica
    torch::Tensor expansion_rbf(torch::Tensor d_ij) {
        double r_c = cutoff_radius; 
        auto cos_cutoff = 0.5 * (torch::cos(M_PI * d_ij / r_c) + 1.0);
        cos_cutoff = torch::where(d_ij > r_c, torch::zeros_like(cos_cutoff), cos_cutoff);

        auto centers = torch::linspace(0.0, r_c, num_radial_basis, d_ij.options());
        double sigma = r_c / num_radial_basis; // Larghezza dipendente dinamicamente dalla scala
        auto rbf = torch::exp(-torch::pow(d_ij.unsqueeze(1) - centers, 2) / torch::pow(torch::full_like(centers, sigma), 2));
        return rbf * cos_cutoff.unsqueeze(1);
    }
    
    // --- FORWARD COMPATIBILE CON PAINN.CPP (TRAINING) ---
    template <typename BatchType>
    torch::Tensor forward(BatchType& batch) {
        auto row = batch.edge_index[0], col = batch.edge_index[1]; 
        auto r_ij = batch.coordinates.index({col}) - batch.coordinates.index({row});
        auto d_ij = torch::sqrt(torch::sum(r_ij * r_ij, 1) + 1e-8);

        torch::Tensor s = embedding->forward(batch.atomic_numbers);
        torch::Tensor v = torch::zeros({s.size(0), 3, s.size(1)}, s.options());

        auto r_ij_norm = r_ij / d_ij.unsqueeze(1);
        auto rbf = expansion_rbf(d_ij);

        // Corretto bug di parsing del template sostituendo lo structured binding
        for (int i = 0; i < num_layers; ++i) {
            auto msg_out = messages[i]->forward(s, v, batch.edge_index, rbf, r_ij_norm);
            s = s + msg_out.first; 
            v = v + msg_out.second;
            
            std::tie(s, v) = updates[i]->forward(s, v);
        }

        torch::Tensor atom_energies = readout->forward(s); 
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

        torch::Tensor atom_energies = readout->forward(s); 
        
        int64_t num_molecules = batch_indices.max().cpu().item<int64_t>() + 1;
        torch::Tensor pred_energy = torch::zeros({num_molecules, 1}, s.options());
        pred_energy.index_add_(0, batch_indices, atom_energies);
        
        return pred_energy.squeeze(-1); 
    }
};
TORCH_MODULE(PaiNNModel);
