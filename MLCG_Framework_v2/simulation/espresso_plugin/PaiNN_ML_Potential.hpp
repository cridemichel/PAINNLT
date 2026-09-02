#pragma once

#include <cstdint>
#include <memory>
#include <string>

// ESPResSo includes
#include "Particle.hpp"
#include "cells.hpp"
#include "nonbonded_interactions/VerletCriterion.hpp"

// PaiNN includes
#include "PaiNN_Architecture.hpp"
#include <torch/torch.h>

class PaiNN_ML_Potential {
public:
    PaiNN_ML_Potential(
        const std::string& model_path,
        int num_species,
        int hidden_channels,
        int n_layers,
        int num_rbf,
        double cutoff,
        double toxvaerd_alpha,
        int ordered_geometry_nodes = 0,
        int ordered_geometry_head_layers = 0,
        int ordered_geometry_head_width = 0,
        double ordered_geometry_energy_scale_kj_mol = 0.0,
        const std::string& device_str = "auto",
        const std::string& precision_str = "float32");

    // Evaluates the ML potential and adds forces to particles
    void calculate_forces(CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion);

    double get_cutoff() const { return m_cutoff; }

    // Ritorna l'ultima energia potenziale calcolata dal modello
    double get_last_energy() const { return m_last_energy; }

    // Profiling is opt-in and disabled by default. It must never alter the
    // graph, Hamiltonian, precision, or force accumulation path.
    void configure_profiling(bool enabled, std::int64_t warmup_calls = 0);
    void reset_profiling();
    std::string get_profile_json() const;

private:
    void calculate_forces_impl(
        CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion);

    PaiNNModel model{nullptr};
    double m_cutoff;
    int m_num_species;
    double m_last_energy = 0.0;
    torch::Device m_device{torch::kCPU};
    torch::Dtype m_dtype{torch::kFloat32};
    std::int64_t m_mps_empty_cache_every_force_calls = 0;
    std::int64_t m_successful_force_calls = 0;

    struct ProfileAccumulator {
        bool enabled = false;
        std::int64_t warmup_calls = 0;
        std::int64_t total_calls = 0;
        std::int64_t measured_calls = 0;
        double total_ms = 0.0;
        double node_index_ms = 0.0;
        double neighbor_traversal_ms = 0.0;
        double edge_pack_ms = 0.0;
        double tensor_inputs_ms = 0.0;
        double forward_ms = 0.0;
        double energy_scalar_ms = 0.0;
        double autograd_ms = 0.0;
        double force_to_cpu_ms = 0.0;
        double force_scatter_ms = 0.0;
        double particles_sum = 0.0;
        double directed_edges_sum = 0.0;
        double physical_pairs_sum = 0.0;
        double host_payload_lower_bound_bytes_sum = 0.0;
        std::int64_t particles_max = 0;
        std::int64_t directed_edges_max = 0;
        std::int64_t physical_pairs_max = 0;
    };

    ProfileAccumulator m_profile;
};

// Global instance to be used in integrate.cpp or forces.cpp
extern std::shared_ptr<PaiNN_ML_Potential> global_painn_potential;
