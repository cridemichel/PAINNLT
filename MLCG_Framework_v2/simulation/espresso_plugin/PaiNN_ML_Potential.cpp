#include "PaiNN_ML_Potential.hpp"
#include "Particle.hpp"
#include "cells.hpp"
#include "exclusions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::shared_ptr<PaiNN_ML_Potential> global_painn_potential = nullptr;

namespace {

torch::Tensor sum_atom_energies_for_hamiltonian(torch::Tensor const &atom_energies) {
    // CPU supports a float64 accumulator, which substantially reduces loss of
    // significance.  Apple MPS does not support float64 tensors, so after the
    // isolated-species gauge has removed the large constant offset we retain
    // the native float32 scalar there.  Both branches remain part of the same
    // autograd graph used for forces and reported energy.
    if (atom_energies.device().is_cpu()) {
        return atom_energies.to(torch::kFloat64).sum();
    }
    return atom_energies.sum();
}

} // namespace

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

        // Report the raw, unconstrained isolated-species offsets.  They are
        // subtracted inside every forward pass by the fixed energy gauge and
        // therefore never contaminate the logged Hamiltonian.
        {
            torch::NoGradGuard no_grad;
            auto species = torch::arange(
                m_num_species,
                torch::TensorOptions().dtype(torch::kInt64).device(m_device));
            auto references = model->isolated_species_reference_table(species)
                                  .squeeze(-1)
                                  .to(torch::kCPU)
                                  .to(torch::kFloat64);
            const double min_reference = references.min().item<double>();
            const double max_reference = references.max().item<double>();
            const double max_abs_reference = references.abs().max().item<double>();
            std::cout << "[PaiNN] Energy gauge: isolated_species_zero_v1 "
                      << "(raw offsets min=" << min_reference
                      << ", max=" << max_reference
                      << ", max_abs=" << max_abs_reference << ")\n";
        }
        
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

    // Cell and Verlet traversal order can change after neighbour-list rebuilds.
    // Assign graph-node indices from particle ids instead, so the same physical
    // configuration always produces the same tensor layout.
    std::vector<Particle*> local_ml_particles;
    auto local_particles = cell_structure.local_particles();
    for (auto& p : local_particles) {
        if (p.type() < m_num_species) {
            local_ml_particles.push_back(&p);
        }
    }
    std::sort(
        local_ml_particles.begin(), local_ml_particles.end(),
        [](Particle const* lhs, Particle const* rhs) { return lhs->id() < rhs->id(); });

    idx_to_particle.reserve(local_ml_particles.size());
    atomic_numbers.reserve(local_ml_particles.size());
    for (auto* particle : local_ml_particles) {
        const int index = static_cast<int>(idx_to_particle.size());
        const auto inserted = pid_to_idx.emplace(particle->id(), index);
        if (!inserted.second) {
            throw std::runtime_error(
                "PaiNN found duplicate local particle id " + std::to_string(particle->id()));
        }
        idx_to_particle.push_back(particle);
        atomic_numbers.push_back(particle->type());
    }

    const int num_particles = static_cast<int>(idx_to_particle.size());
    if (num_particles == 0) {
        return;
    }

    using PairKey = std::pair<int, int>;
    using Displacement = std::array<float, 3>;
    std::map<PairKey, Displacement> physical_pairs;

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

        // Store each physical pair once in a canonical order.  Periodic ghost
        // aliases may expose the same pair more than once; duplicate traversal
        // must not duplicate the interaction energy or force.
        const int low = std::min(idx1, idx2);
        const int high = std::max(idx1, idx2);
        Displacement r_low_minus_high{};
        if (idx1 == low) {
            r_low_minus_high = {
                static_cast<float>(d.vec21[0]),
                static_cast<float>(d.vec21[1]),
                static_cast<float>(d.vec21[2])};
        } else {
            r_low_minus_high = {
                static_cast<float>(-d.vec21[0]),
                static_cast<float>(-d.vec21[1]),
                static_cast<float>(-d.vec21[2])};
        }

        const auto [it, inserted] = physical_pairs.emplace(
            PairKey{low, high}, r_low_minus_high);
        if (!inserted) {
            double squared_difference = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double difference =
                    static_cast<double>(it->second[axis]) - r_low_minus_high[axis];
                squared_difference += difference * difference;
            }
            if (squared_difference > 1.0e-12) {
                throw std::runtime_error(
                    "PaiNN encountered inconsistent periodic images for the same physical pair. "
                    "Increase the box or reduce cutoff+skin.");
            }
        }
    };

    cell_structure.non_bonded_loop(painn_kernel, verlet_criterion);

    std::vector<int64_t> edge_rows;
    std::vector<int64_t> edge_cols;
    std::vector<float> r_ij_data;
    edge_rows.reserve(2 * physical_pairs.size());
    edge_cols.reserve(2 * physical_pairs.size());
    r_ij_data.reserve(6 * physical_pairs.size());
    for (const auto& [pair, r_low_minus_high] : physical_pairs) {
        const int low = pair.first;
        const int high = pair.second;

        edge_rows.push_back(low);
        edge_cols.push_back(high);
        r_ij_data.insert(
            r_ij_data.end(),
            {r_low_minus_high[0], r_low_minus_high[1], r_low_minus_high[2]});

        edge_rows.push_back(high);
        edge_cols.push_back(low);
        r_ij_data.insert(
            r_ij_data.end(),
            {-r_low_minus_high[0], -r_low_minus_high[1], -r_low_minus_high[2]});
    }

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

        // The isolated-species gauge makes this energy exactly zero while
        // retaining a complete forward path and exactly zero forces.
        const torch::Tensor atom_energies =
            model->forward_atom_energies(t_atomic_numbers, t_r_ij, t_edge_index)
                .squeeze(-1);
        m_last_energy = sum_atom_energies_for_hamiltonian(atom_energies).item<double>();
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
    const torch::Tensor total_energy = sum_atom_energies_for_hamiltonian(atom_energies);

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
