#pragma once
#include <torch/torch.h>
#include <cmath>
#include <vector>

// ============================================================================
// 1. BLOCCO MESSAGGI (Message Passing)
// ============================================================================
struct PaiNNMessageImpl : torch::nn::Module {
    torch::nn::Linear scalar_mlp{nullptr}, filter_mlp{nullptr};
    PaiNNMessageImpl(int dim) {
        scalar_mlp = register_module("scalar_mlp", torch::nn::Linear(dim, dim * 3));
        filter_mlp = register_module("filter_mlp", torch::nn::Linear(20, dim * 3)); // 20 num_rbf hardcoded in painn.cpp
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

    PaiNNModelImpl(int num_embeddings, int dim, int layers, int num_rbf = 20, double cutoff = 5.0) 
        : num_layers(layers) {
        
        embedding = register_module("embedding", torch::nn::Embedding(num_embeddings, dim));
        for (int i = 0; i < layers; ++i) {
            messages.push_back(register_module("message_" + std::to_string(i), PaiNNMessage(dim)));
            updates.push_back(register_module("update_" + std::to_string(i), PaiNNUpdate(dim)));
        }
        readout = register_module("readout", torch::nn::Sequential(
            torch::nn::Linear(dim, dim / 2), torch::nn::SiLU(), torch::nn::Linear(dim / 2, 1)
        ));
    }

    // Identica espansione RBF del file painn.cpp
    torch::Tensor expansion_rbf(torch::Tensor d_ij) {
        auto centers = torch::linspace(0.0, 5.0, 20, d_ij.options());
        return torch::exp(-torch::pow(d_ij.unsqueeze(1) - centers, 2) / torch::pow(torch::full_like(centers, 0.5), 2));
    }

    // --- FORWARD CON R_IJ ESPLICITO (PER ESPRESSO PBC) ---
    torch::Tensor forward_with_rij(torch::Tensor atomic_numbers, 
                                   torch::Tensor r_ij,
                                   torch::Tensor edge_index, 
                                   torch::Tensor batch_indices) {
        
        auto d_ij = torch::norm(r_ij, 2, 1) + 1e-8; // 1e-8 evita divisioni per zero
        
        // Inizializzazione Scalari e Vettori
        torch::Tensor s = embedding->forward(atomic_numbers);
        torch::Tensor v = torch::zeros({s.size(0), 3, s.size(1)}, s.options());

        // Geometria
        auto r_ij_norm = r_ij / d_ij.unsqueeze(1);
        auto rbf = expansion_rbf(d_ij);

        // Ciclo dei layer
        for (int i = 0; i < num_layers; ++i) {
            auto [ds, dv] = messages[i]->forward(s, v, edge_index, rbf, r_ij_norm);
            s = s + ds; 
            v = v + dv;
            
            std::tie(s, v) = updates[i]->forward(s, v);
        }

        // Predizione dell'energia atomica
        torch::Tensor atom_energies = readout->forward(s); 
        
        // Somma delle energie per molecola usando batch_indices
        int64_t num_molecules = batch_indices.max().item<int64_t>() + 1;
        torch::Tensor pred_energy = torch::zeros({num_molecules, 1}, s.options());
        pred_energy.index_add_(0, batch_indices, atom_energies);
        
        return pred_energy.squeeze(-1); 
    }
};
TORCH_MODULE(PaiNNModel);
