#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
        bool ordered_geometry_head_only = false,
        int ordered_geometry_copies = 1,
        bool tel22_shared_geometry = false,
        const std::string& device_str = "auto",
        const std::string& precision_str = "float32");

    // Evaluates the ML potential and adds forces to particles
    void calculate_forces(CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion);

    double get_cutoff() const { return m_cutoff; }

    // Ritorna l'ultima energia potenziale calcolata dal modello.
    // Nel percorso "device" l'energia resta un tensore sul device e viene
    // convertita in double solo qui (niente sync/.item() a ogni passo).
    double get_last_energy() const;

    // Graph path selected at construction from MLCG_PAINN_GRAPH
    // ("legacy" = default, ESPResSo pair loop + host graph; "device" =
    // device-resident pair search, node forces from autograd w.r.t. positions).
    bool uses_device_graph() const { return m_device_graph; }

    // Profiling is opt-in and disabled by default. It must never alter the
    // graph, Hamiltonian, precision, or force accumulation path.
    void configure_profiling(bool enabled, std::int64_t warmup_calls = 0);
    void reset_profiling();
    std::string get_profile_json() const;

private:
    void calculate_forces_impl(
        CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion);
    void calculate_forces_device_graph(CellStructure& cell_structure);
    void rebuild_device_graph_cache(
        CellStructure& cell_structure, std::vector<Particle*> const& particles);
    void validate_tel22_layout(
        std::vector<Particle*> const& particles,
        std::vector<int64_t> const& atomic_numbers);

    PaiNNModel model{nullptr};
    double m_cutoff;
    int m_num_species;
    bool m_tel22_shared_geometry = false;
    bool m_tel22_layout_validated = false;
    mutable double m_last_energy = 0.0;
    mutable bool m_energy_pending = false;
    mutable torch::Tensor m_energy_tensor;

    // Device-graph path (MLCG_PAINN_GRAPH=device).  The candidate list holds
    // every intermolecular ML pair i<j once (static topology, single rank);
    // each step only positions travel host->device and node forces back.
    bool m_device_graph = false;
    std::vector<int> m_graph_ids;
    torch::Tensor m_graph_species;   // [N] int64, device
    torch::Tensor m_pair_i;          // [P] int64, device, i<j
    torch::Tensor m_pair_j;          // [P] int64, device
    torch::Tensor m_pos_host;        // [N,3] float64, host (pageable)
    torch::Tensor m_box_t;           // [1,3] float64, device
    torch::Tensor m_box_inv_t;       // [1,3] float64, device (0 = non periodic)
    std::array<double, 3> m_box_cached{{-1.0, -1.0, -1.0}};
    std::array<bool, 3> m_periodic_cached{{false, false, false}};
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
