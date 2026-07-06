#pragma once
#include <torch/torch.h>
#include <cmath>
#include <vector>

// ============================================================================
// 1. BLOCCO INTERAZIONE SCHNET (Continuous Filter Convolution)
// ============================================================================
struct SchNetInteractionImpl : torch::nn::Module {
    torch::nn::Sequential filter_mlp{nullptr};
    torch::nn::Linear in2f{nullptr};
    torch::nn::Sequential f2out{nullptr};
    
    SchNetInteractionImpl(int dim, int num_rbf) {
        filter_mlp = register_module("filter_mlp", torch::nn::Sequential(
            torch::nn::Linear(num_rbf, dim),
            torch::nn::SiLU(),
            torch::nn::Linear(dim, dim)
        ));
        in2f = register_module("in2f", torch::nn::Linear(dim, dim));
        f2out = register_module("f2out", torch::nn::Sequential(
            torch::nn::Linear(dim, dim),
            torch::nn::SiLU(),
            torch::nn::Linear(dim, dim)
        ));
    }
    
    torch::Tensor forward(torch::Tensor s, torch::Tensor edge_index, torch::Tensor rbf) {
        auto row = edge_index[0], col = edge_index[1];
        
        // 1. Calcolo dei pesi dal filtro RBF
        auto w = filter_mlp->forward(rbf);
        
        // 2. Proiezione dell'input
        auto s_row = in2f->forward(s.index({row}));
        
        // 3. Convoluzione continua (moltiplicazione element-wise)
        auto interaction = s_row * w;
        
        // 4. Aggregazione spaziale (somma sui vicini)
        auto delta_s = torch::zeros_like(s);
        delta_s.index_add_(0, col, interaction);
        
        // 5. Update finale
        return s + f2out->forward(delta_s);
    }
};
TORCH_MODULE(SchNetInteraction);

// ============================================================================
// 2. MODELLO COMPLETO SCHNET
// ============================================================================
struct SchNetModelImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    std::vector<SchNetInteraction> interactions;
    torch::nn::Sequential readout{nullptr};
    int num_layers;
    double cutoff_radius; 
    int num_radial_basis; 
    
    SchNetModelImpl(int num_embeddings, int dim, int layers, int num_rbf = 20, double cutoff = 5.0) 
        : num_layers(layers), cutoff_radius(cutoff), num_radial_basis(num_rbf) {
        
        embedding = register_module("embedding", torch::nn::Embedding(num_embeddings, dim));
        
        for (int i = 0; i < layers; ++i) {
            interactions.push_back(register_module("interaction_" + std::to_string(i), SchNetInteraction(dim, num_rbf)));
        }
        
        readout = register_module("readout", torch::nn::Sequential(
            torch::nn::Linear(dim, dim / 2), 
            torch::nn::SiLU(), 
            torch::nn::Linear(dim / 2, 1)
        ));
    }

    // Espansione RBF con Cosine Cutoff integrato
    torch::Tensor expansion_rbf(torch::Tensor d_ij) {
        double r_c = cutoff_radius; 
        auto cos_cutoff = 0.5 * (torch::cos(M_PI * d_ij / r_c) + 1.0);
        cos_cutoff = torch::where(d_ij > r_c, torch::zeros_like(cos_cutoff), cos_cutoff);

        auto centers = torch::linspace(0.0, r_c, num_radial_basis, d_ij.options());
        double sigma = r_c / num_radial_basis;
        auto rbf = torch::exp(-torch::pow(d_ij.unsqueeze(1) - centers, 2) / torch::pow(torch::full_like(centers, sigma), 2));
        return rbf * cos_cutoff.unsqueeze(1);
    }
    
    // --- FORWARD COMPATIBILE (TRAINING) ---
    template <typename BatchType>
    torch::Tensor forward(BatchType& batch) {
        auto row = batch.edge_index[0], col = batch.edge_index[1]; 
        auto r_ij = batch.coordinates.index({col}) - batch.coordinates.index({row});
        auto d_ij = torch::norm(r_ij, 2, 1) + 1e-8; 

        torch::Tensor s = embedding->forward(batch.atomic_numbers);
        auto rbf = expansion_rbf(d_ij);

        for (int i = 0; i < num_layers; ++i) {
            s = interactions[i]->forward(s, batch.edge_index, rbf);
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
        
        auto d_ij = torch::norm(r_ij, 2, 1) + 1e-8; 
        
        torch::Tensor s = embedding->forward(atomic_numbers);
        auto rbf = expansion_rbf(d_ij);

        for (int i = 0; i < num_layers; ++i) {
            s = interactions[i]->forward(s, edge_index, rbf);
        }

        torch::Tensor atom_energies = readout->forward(s); 
        
        int64_t num_molecules = batch_indices.max().cpu().item<int64_t>() + 1;
        torch::Tensor pred_energy = torch::zeros({num_molecules, 1}, s.options());
        pred_energy.index_add_(0, batch_indices, atom_energies);
        
        return pred_energy.squeeze(-1); 
    }
};
TORCH_MODULE(SchNetModel);
