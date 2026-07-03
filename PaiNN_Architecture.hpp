#pragma once
#include <torch/torch.h>
#include <cmath>

// ============================================================================
// 1. ESPANSIONE DELLE DISTANZE (Radial Basis Function - RBF)
// ============================================================================
struct GaussianRBFImpl : torch::nn::Module {
    torch::Tensor centers;
    double gamma;

    GaussianRBFImpl(int num_rbf = 20, double cutoff = 5.0) {
        // Centri equispaziati tra 0 e cutoff
        auto c = torch::linspace(0.0, cutoff, num_rbf);
        centers = register_buffer("centers", c);
        // Parametro di larghezza della Gaussiana
        gamma = 1.0 / std::pow((cutoff / num_rbf), 2); 
    }

    torch::Tensor forward(torch::Tensor distances) {
        return torch::exp(-gamma * torch::pow(distances.unsqueeze(-1) - centers, 2));
    }
};
TORCH_MODULE(GaussianRBF);

// ============================================================================
// 2. BLOCCO MESSAGGI (Message Passing)
// ============================================================================
struct PaiNNMessageImpl : torch::nn::Module {
    torch::nn::Linear scalar_proj{nullptr};
    torch::nn::Linear filter_net{nullptr};
    int hidden_channels;

    PaiNNMessageImpl(int hidden, int num_rbf) : hidden_channels(hidden) {
        // Proiezione degli scalari (espande a 3 * hidden_channels)
        scalar_proj = register_module("scalar_proj", torch::nn::Linear(hidden, 3 * hidden));
        // Rete filtro per le RBF
        filter_net = register_module("filter_net", torch::nn::Linear(num_rbf, 3 * hidden));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor s, torch::Tensor v, torch::Tensor edge_index, 
        torch::Tensor rbf, torch::Tensor r_ij_norm) {
        
        auto row = edge_index[0]; // Nodi sorgente
        auto col = edge_index[1]; // Nodi destinazione

        // Calcola il filtro geometrico dalle distanze
        auto filter = filter_net(rbf);
        
        // Proietta le feature scalari dei vicini
        auto s_j = scalar_proj(s.index({col}));
        
        // Moltiplica le feature scalari per il filtro (Gating)
        auto gated_s = s_j * filter;

        // Separa il tensore in 3 parti (ognuna di dimensione 'hidden_channels')
        auto chunks = gated_s.chunk(3, -1);
        auto s_val = chunks[0];
        auto v_val1 = chunks[1];
        auto v_val2 = chunks[2];

        // Aggiornamento dei Vettori (tiene conto della direzionalità r_ij_norm)
        auto v_j = v.index({col});
        auto dv_message = v_val1.unsqueeze(1) * v_j + v_val2.unsqueeze(1) * r_ij_norm.unsqueeze(-1);
        
        // Aggiornamento degli Scalari
        auto ds_message = s_val;

        // Aggrega i messaggi sui nodi destinazione (Somma)
        auto ds = torch::zeros_like(s);
        auto dv = torch::zeros_like(v);
        ds.index_add_(0, row, ds_message);
        dv.index_add_(0, row, dv_message);

        return {ds, dv};
    }
};
TORCH_MODULE(PaiNNMessage);

// ============================================================================
// 3. BLOCCO AGGIORNAMENTO (Update Block)
// ============================================================================
struct PaiNNUpdateImpl : torch::nn::Module {
    torch::nn::Linear U{nullptr}, V{nullptr}, mlp_proj1{nullptr}, mlp_proj2{nullptr};

    PaiNNUpdateImpl(int hidden) {
        U = register_module("U", torch::nn::Linear(hidden, hidden));
        V = register_module("V", torch::nn::Linear(hidden, hidden));
        mlp_proj1 = register_module("mlp_proj1", torch::nn::Linear(hidden * 2, hidden));
        mlp_proj2 = register_module("mlp_proj2", torch::nn::Linear(hidden, hidden));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor s, torch::Tensor v) {
        // Combinazione lineare dei vettori sui canali (l'ultima dimensione è già 128)
        auto u_v = U(v);
        auto v_v = V(v);
       
        // Norma dei vettori per iniettare l'informazione geometrica negli scalari
        auto v_norm = torch::norm(v_v, 2, 1);
        
        // MLP per aggiornare le feature scalari
        auto s_in = torch::cat({s, v_norm}, -1);
        auto ds = mlp_proj2(torch::silu(mlp_proj1(s_in)));
        
        // Aggiornamento vettori
        auto dv = u_v * ds.unsqueeze(1);

        return {ds, dv};
    }
};
TORCH_MODULE(PaiNNUpdate);

// ============================================================================
// 4. MODELLO COMPLETO (PaiNN)
// ============================================================================
struct PaiNNModelImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    GaussianRBF expansion_rbf{nullptr};
    torch::nn::ModuleList messages{nullptr};
    torch::nn::ModuleList updates{nullptr};
    torch::nn::Sequential readout{nullptr};
    
    int num_layers;

    PaiNNModelImpl(int num_atoms, int hidden_channels, int n_layers, int num_rbf = 20, double cutoff = 5.0) 
        : num_layers(n_layers) {
        
        // 1. Embedding atomico
        embedding = register_module("embedding", torch::nn::Embedding(num_atoms, hidden_channels));
        
        // 2. RBF
        expansion_rbf = register_module("expansion_rbf", GaussianRBF(num_rbf, cutoff));
        
        // 3. Moduli di Message Passing e Update
        messages = register_module("messages", torch::nn::ModuleList());
        updates = register_module("updates", torch::nn::ModuleList());
        
        for (int i = 0; i < num_layers; ++i) {
            messages->push_back(PaiNNMessage(hidden_channels, num_rbf));
            updates->push_back(PaiNNUpdate(hidden_channels));
        }

        // 4. Readout finale (da scalari a Energia)
        readout = register_module("readout", torch::nn::Sequential(
            torch::nn::Linear(hidden_channels, hidden_channels / 2),
            torch::nn::SiLU(),
            torch::nn::Linear(hidden_channels / 2, 1)
        ));
    }

    // --- FORWARD SLEGATO DA STRUTTURE CUSTOM (PERFETTO PER ESPRESSO) ---
    torch::Tensor forward(torch::Tensor atomic_numbers, 
                          torch::Tensor coordinates, 
                          torch::Tensor edge_index, 
                          torch::Tensor batch_indices) {
        
        auto row = edge_index[0];
        auto col = edge_index[1]; 
        
        // Vettori distanza (r_ij) e loro norma (d_ij)
        auto r_ij = coordinates.index({col}) - coordinates.index({row});
        auto d_ij = torch::norm(r_ij, 2, 1) + 1e-8; // 1e-8 evita divisioni per zero
        
        // Inizializzazione Scalari e Vettori
        torch::Tensor s = embedding(atomic_numbers);
        torch::Tensor v = torch::zeros({s.size(0), 3, s.size(1)}, s.options());

        // Geometria
        auto r_ij_norm = r_ij / d_ij.unsqueeze(1);
        auto rbf = expansion_rbf(d_ij);

        // Ciclo dei layer
        for (int i = 0; i < num_layers; ++i) {
            auto msg_out = messages[i]->as<PaiNNMessage>()->forward(s, v, edge_index, rbf, r_ij_norm);
            s = s + std::get<0>(msg_out);
            v = v + std::get<1>(msg_out);
            
            auto upd_out = updates[i]->as<PaiNNUpdate>()->forward(s, v);
            s = s + std::get<0>(upd_out);
            v = v + std::get<1>(upd_out);
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
