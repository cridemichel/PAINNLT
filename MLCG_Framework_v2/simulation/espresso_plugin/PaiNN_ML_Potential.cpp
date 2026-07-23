#include "PaiNN_ML_Potential.hpp"
#include "Particle.hpp"
#include "cells.hpp"

#include <iostream>
#include <vector>
#include <unordered_map>

std::shared_ptr<PaiNN_ML_Potential> global_painn_potential = nullptr;

PaiNN_ML_Potential::PaiNN_ML_Potential(const std::string& model_path, int num_species, int hidden_channels, int n_layers, int num_rbf, double cutoff, double toxvaerd_alpha, const std::string& device_str) 
    : m_cutoff(cutoff), m_num_species(num_species) {
    
    // Inizializza il modello C++ con i parametri di architettura
    model = PaiNNModel(num_species, hidden_channels, n_layers, num_rbf, cutoff, toxvaerd_alpha);
    
    // Carica i pesi dal file .pt salvato durante il training
    try {
        torch::load(model, model_path);
        model->eval(); // Mette il modello in modalità inferenza
        for (auto& param : model->parameters()) {
            param.set_requires_grad(false);
        }
        
        // Rilevamento Device
        if (device_str == "cuda" && torch::cuda::is_available()) {
            m_device = torch::Device(torch::kCUDA);
            std::cout << "[PaiNN] Accelerazione GPU (CUDA) forzata!\n";
        } else if (device_str == "mps" && torch::mps::is_available()) {
            m_device = torch::Device(torch::kMPS);
            std::cout << "[PaiNN] Accelerazione GPU (MPS) forzata!\n";
        } else if (device_str == "cpu") {
            m_device = torch::Device(torch::kCPU);
            std::cout << "[PaiNN] Esecuzione su CPU forzata.\n";
        } else {
            // Auto-detect
            if (torch::cuda::is_available()) {
                m_device = torch::Device(torch::kCUDA);
                std::cout << "[PaiNN] Accelerazione GPU (CUDA) attivata (Auto)!\n";
            } else if (torch::mps::is_available()) {
                m_device = torch::Device(torch::kMPS);
                std::cout << "[PaiNN] Accelerazione GPU (MPS) attivata (Auto)!\n";
            } else {
                std::cout << "[PaiNN] GPU non trovata o device_str invalido. Esecuzione su CPU (Auto).\n";
            }
        }
        model->to(m_device);
        
        std::cout << "[PaiNN] Modello C++ inizializzato e pesi caricati da: " << model_path << "\n";
    } catch (const c10::Error& e) {
        std::cerr << "[PaiNN] Errore nel caricamento del modello: " << e.what() << "\n";
    }
}

void PaiNN_ML_Potential::calculate_forces(CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion) {
    // 1. Mappatura delle particelle attuali a indici contigui per i tensori PyTorch
    std::unordered_map<int, int> pid_to_idx;
    std::vector<Particle*> idx_to_particle;
    std::vector<int64_t> atomic_numbers;
    
    int current_idx = 0;
    
    // Raccogliamo tutte le particelle locali che appartengono al modello ML
    auto local_particles = cell_structure.local_particles();
    for (auto& p : local_particles) {
        if (p.type() >= m_num_species) continue;
        pid_to_idx[p.id()] = current_idx++;
        idx_to_particle.push_back(&p);
        atomic_numbers.push_back(p.type()); // Assumiamo che il 'type' di ESPResSo sia l'atomic number
    }
    
    int num_local_ml_particles = current_idx;
    
    // Raccogliamo anche le particelle ghost per le interazioni di bordo (PBC)
    auto ghost_particles = cell_structure.ghost_particles();
    for (auto& p : ghost_particles) {
        if (p.type() >= m_num_species) continue;
        pid_to_idx[p.id()] = current_idx++;
        idx_to_particle.push_back(&p);
        atomic_numbers.push_back(p.type());
    }
    
    int total_particles = current_idx;
    if (total_particles == 0) return;

    // 2. Costruzione del grafo usando il neighbor list di ESPResSo
    std::vector<int64_t> edge_rows;
    std::vector<int64_t> edge_cols;
    std::vector<float> r_ij_data;
    
    auto painn_kernel = [&](Particle const &p1, Particle const &p2, Distance const &d) {
        if (p1.type() >= m_num_species || p2.type() >= m_num_species) return; // Filtro particelle non ML
        if (d.dist2 > m_cutoff * m_cutoff) return; // Filtro cutoff
        
        int idx1 = pid_to_idx[p1.id()];
        int idx2 = pid_to_idx[p2.id()];
        
        // In ESPResSo, d.vec21 è il vettore da p2 a p1 (pos1 - pos2).
        // Nel training, r_ij = pos_row - pos_col.
        // Se row = p2 (idx2) e col = p1 (idx1), allora r_ij = pos2 - pos1 = -d.vec21.
        
        // Arco p1 -> p2 (col=p1, row=p2)
        edge_rows.push_back(idx2);
        edge_cols.push_back(idx1);
        r_ij_data.push_back(static_cast<float>(-d.vec21[0]));
        r_ij_data.push_back(static_cast<float>(-d.vec21[1]));
        r_ij_data.push_back(static_cast<float>(-d.vec21[2]));
        
        // Arco p2 -> p1 (col=p2, row=p1)
        // r_ij = pos1 - pos2 = +d.vec21
        edge_rows.push_back(idx1);
        edge_cols.push_back(idx2);
        r_ij_data.push_back(static_cast<float>(d.vec21[0]));
        r_ij_data.push_back(static_cast<float>(d.vec21[1]));
        r_ij_data.push_back(static_cast<float>(d.vec21[2]));
    };

    // Esegue il loop di ESPResSo su tutte le coppie di vicini
    cell_structure.non_bonded_loop(painn_kernel, verlet_criterion);
    
    int num_edges = edge_rows.size();
    if (num_edges == 0) return; // Nessuna interazione

    // 3. Creazione dei Tensori
    torch::Tensor t_atomic_numbers = torch::tensor(atomic_numbers, torch::kInt64).to(m_device);
    
    std::vector<int64_t> flat_edges;
    flat_edges.insert(flat_edges.end(), edge_rows.begin(), edge_rows.end());
    flat_edges.insert(flat_edges.end(), edge_cols.begin(), edge_cols.end());
    torch::Tensor t_edge_index = torch::tensor(flat_edges, torch::kInt64).reshape({2, num_edges}).to(m_device);
    
    torch::Tensor t_r_ij = torch::tensor(r_ij_data, torch::kFloat32).reshape({num_edges, 3}).to(m_device);
    t_r_ij.set_requires_grad(true); // Fondamentale: calcoliamo i gradienti rispetto ai vettori distanza!

    torch::Tensor t_batch = torch::zeros({total_particles}, torch::kInt64).to(m_device);

    // 4. Inferenza del Modello
    torch::Tensor energy = model->forward_with_rij(t_atomic_numbers, t_r_ij, t_edge_index, t_batch);
    
    // CORREZIONE BUG GHOST PARTICLES:
    // Sommiamo solo l'energia atomica calcolata per le particelle *reali* (locali), ignorando i ghost.
    m_last_energy = energy.slice(0, 0, num_local_ml_particles).sum().item<double>();
    
    // 5. Calcolo delle Forze (Gradients w.r.t r_ij)
    auto grads = torch::autograd::grad({energy.sum()}, {t_r_ij}, {torch::ones_like(energy.sum())}, false, false);
    torch::Tensor f_r_ij = grads[0].cpu(); // Riportiamo i gradienti su CPU per assegnarli a ESPResSo

    // 6. Assegnazione delle Forze alle Particelle ESPResSo
    // Per ogni arco col->row (dove r_ij = r_col - r_row), la forza associata a r_ij è f_r_ij.
    // Forza su col: -f_r_ij
    // Forza su row: +f_r_ij
    auto f_r_ij_acc = f_r_ij.accessor<float, 2>();
    
    for (int e = 0; e < num_edges; ++e) {
        int r = edge_rows[e]; // row
        int c = edge_cols[e]; // col
        
        float fx = f_r_ij_acc[e][0];
        float fy = f_r_ij_acc[e][1];
        float fz = f_r_ij_acc[e][2];
        
        // Assegniamo le forze (ESPResSo gestirà la comunicazione delle forze dei ghost)
        // La forza è la derivata negativa dell'energia rispetto alla posizione.
        // F_row = - dE / d(pos_row) = - dE / dr_ij = -f_r_ij
        // F_col = - dE / d(pos_col) = + dE / dr_ij = +f_r_ij
        idx_to_particle[r]->force()[0] -= fx;
        idx_to_particle[r]->force()[1] -= fy;
        idx_to_particle[r]->force()[2] -= fz;
        
        idx_to_particle[c]->force()[0] += fx;
        idx_to_particle[c]->force()[1] += fy;
        idx_to_particle[c]->force()[2] += fz;
    }
}
