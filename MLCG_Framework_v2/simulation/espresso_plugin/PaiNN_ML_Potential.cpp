#include "PaiNN_ML_Potential.hpp"
#include "Particle.hpp"
#include "cells.hpp"
#include "exclusions.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
        throw;
    }
}

void PaiNN_ML_Potential::calculate_forces(CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion) {
    // Never expose an energy value from a previous integration step.
    m_last_energy = 0.0;

    // The production path is deliberately single-rank.  Each physical ML site
    // is represented exactly once, by its local particle.  Periodic ghost
    // copies are only aliases used by ESPResSo's neighbour loop and must never
    // become independent PaiNN nodes or independent atomic-energy terms.
    std::unordered_map<int, int> pid_to_idx;
    std::vector<Particle*> idx_to_particle;
    std::vector<int64_t> atomic_numbers;

    auto local_particles = cell_structure.local_particles();
    for (auto& p : local_particles) {
        if (p.type() >= m_num_species) {
            continue;
        }
        const int index = static_cast<int>(idx_to_particle.size());
        const auto inserted = pid_to_idx.emplace(p.id(), index);
        if (!inserted.second) {
            throw std::runtime_error(
                "PaiNN found duplicate local particle id " + std::to_string(p.id()));
        }
        idx_to_particle.push_back(&p);
        atomic_numbers.push_back(p.type());
    }

    const int num_particles = static_cast<int>(idx_to_particle.size());
    if (num_particles == 0) {
        return;
    }

    std::vector<int64_t> edge_rows;
    std::vector<int64_t> edge_cols;
    std::vector<float> r_ij_data;

    auto painn_kernel = [&](Particle const& p1, Particle const& p2, Distance const& d) {
        if (p1.type() >= m_num_species || p2.type() >= m_num_species) {
            return;
        }
        if (d.dist2 > m_cutoff * m_cutoff) {
            return;
        }
        if (p1.mol_id() == p2.mol_id()) {
            return;
        }

        // In a one-rank run, a periodic ghost has the same physical particle
        // id as a local site.  Reuse that local node while retaining d.vec21,
        // which already contains ESPResSo's minimum-image displacement.
        const auto found1 = pid_to_idx.find(p1.id());
        const auto found2 = pid_to_idx.find(p2.id());
        if (found1 == pid_to_idx.end() || found2 == pid_to_idx.end()) {
            throw std::runtime_error(
                "PaiNN encountered a neighbour without a local physical node. "
                "This indicates the uncertified multi-rank/halo path; run with one MPI rank.");
        }

        const int idx1 = found1->second;
        const int idx2 = found2->second;
        if (idx1 == idx2) {
            throw std::runtime_error(
                "PaiNN encountered a periodic self-image inside the cutoff. "
                "Increase the box or reduce cutoff+skin.");
        }

        // In ESPResSo, Distance::vec21 is calculated as p1.pos() - p2.pos(),
        // which means d.vec21 is r1 - r2.
        // For edge idx2 -> idx1: row=idx1, col=idx2.
        // PyTorch expects r_ij = r_row - r_col
        
        // Edge 1: row=idx2, col=idx1
        // r_ij = r_2 - r_1 = - (r_1 - r_2) = -d.vec21
        edge_rows.push_back(idx2);
        edge_cols.push_back(idx1);
        r_ij_data.push_back(static_cast<float>(-d.vec21[0]));
        r_ij_data.push_back(static_cast<float>(-d.vec21[1]));
        r_ij_data.push_back(static_cast<float>(-d.vec21[2]));

        // Edge 2: row=idx1, col=idx2
        // r_ij = r_1 - r_2 = d.vec21
        edge_rows.push_back(idx1);
        edge_cols.push_back(idx2);
        r_ij_data.push_back(static_cast<float>(d.vec21[0]));
        r_ij_data.push_back(static_cast<float>(d.vec21[1]));
        r_ij_data.push_back(static_cast<float>(d.vec21[2]));
    };

    cell_structure.non_bonded_loop(painn_kernel, verlet_criterion);

    const int num_edges = static_cast<int>(edge_rows.size());
    torch::Tensor t_atomic_numbers =
        torch::tensor(atomic_numbers, torch::TensorOptions().dtype(torch::kInt64))
            .to(m_device);

    torch::Tensor t_edge_index;
    torch::Tensor t_r_ij;
    if (num_edges == 0) {
        t_edge_index = torch::empty(
            {2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(m_device));
        t_r_ij = torch::empty(
            {0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(m_device));

        // Isolated-site readout energies are part of the trained Hamiltonian.
        // They are position-independent, so forces are exactly zero, but the
        // energy must still be reported instead of being reset to zero.
        const torch::Tensor atom_energies =
            model->forward_atom_energies(t_atomic_numbers, t_r_ij, t_edge_index)
                .squeeze(-1);
        m_last_energy = atom_energies.sum().item<double>();
        return;
    }

    std::vector<int64_t> flat_edges;
    flat_edges.reserve(static_cast<std::size_t>(2 * num_edges));
    flat_edges.insert(flat_edges.end(), edge_rows.begin(), edge_rows.end());
    flat_edges.insert(flat_edges.end(), edge_cols.begin(), edge_cols.end());
    t_edge_index =
        torch::tensor(flat_edges, torch::TensorOptions().dtype(torch::kInt64))
            .reshape({2, num_edges})
            .to(m_device);

    t_r_ij =
        torch::tensor(r_ij_data, torch::TensorOptions().dtype(torch::kFloat32))
            .reshape({num_edges, 3})
            .to(m_device);
    t_r_ij.set_requires_grad(true);

    const torch::Tensor atom_energies =
        model->forward_atom_energies(t_atomic_numbers, t_r_ij, t_edge_index)
            .squeeze(-1);
    const torch::Tensor total_energy = atom_energies.sum();

    // Energy and forces are derived from exactly the same scalar Hamiltonian.
    // There are no ghost atom-energy terms in this single-rank graph.
    m_last_energy = total_energy.item<double>();
    auto grads = torch::autograd::grad(
        {total_energy}, {t_r_ij}, {torch::ones_like(total_energy)}, false, false);
    const torch::Tensor f_r_ij = -grads[0].cpu();
    auto f_r_ij_acc = f_r_ij.accessor<float, 2>();

    for (int e = 0; e < num_edges; ++e) {
        const int row = static_cast<int>(edge_rows[e]);
        const int col = static_cast<int>(edge_cols[e]);
        const float fx = f_r_ij_acc[e][0];
        const float fy = f_r_ij_acc[e][1];
        const float fz = f_r_ij_acc[e][2];

        idx_to_particle[row]->force()[0] += fx;
        idx_to_particle[row]->force()[1] += fy;
        idx_to_particle[row]->force()[2] += fz;

        idx_to_particle[col]->force()[0] -= fx;
        idx_to_particle[col]->force()[1] -= fy;
        idx_to_particle[col]->force()[2] -= fz;
    }
}
