#include "PaiNN_ML_Potential.hpp"
#include "Particle.hpp"
#include "cells.hpp"
#include "exclusions.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <ATen/mps/MPSAllocatorInterface.h>
#endif

std::shared_ptr<PaiNN_ML_Potential> global_painn_potential = nullptr;

namespace {

using ProfileClock = std::chrono::steady_clock;

std::int64_t nonnegative_integer_environment(
    const char* name, std::int64_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    const std::string text(raw);
    std::size_t consumed = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &consumed, 10);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            std::string(name) + " must be a non-negative integer, got '" + text + "'");
    }
    if (consumed != text.size() || value < 0) {
        throw std::invalid_argument(
            std::string(name) + " must be a non-negative integer, got '" + text + "'");
    }
    return static_cast<std::int64_t>(value);
}

double elapsed_ms(ProfileClock::time_point const &start, ProfileClock::time_point const &end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

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

PaiNN_ML_Potential::PaiNN_ML_Potential(
    const std::string& model_path,
    int num_species,
    int hidden_channels,
    int n_layers,
    int num_rbf,
    double cutoff,
    double toxvaerd_alpha,
    int ordered_geometry_nodes,
    int ordered_geometry_head_layers,
    int ordered_geometry_head_width,
    double ordered_geometry_energy_scale_kj_mol,
    bool ordered_geometry_head_only,
    const std::string& device_str,
    const std::string& precision_str)
    : m_cutoff(cutoff), m_num_species(num_species) {
    
    // Inizializza il modello C++ con i parametri di architettura
    model = PaiNNModel(
        num_species,
        hidden_channels,
        n_layers,
        num_rbf,
        cutoff,
        toxvaerd_alpha,
        ordered_geometry_nodes,
        ordered_geometry_head_layers,
        ordered_geometry_head_width,
        ordered_geometry_energy_scale_kj_mol,
        ordered_geometry_head_only);
    
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
        if (precision_str == "float32") {
            m_dtype = torch::kFloat32;
        } else if (precision_str == "float64") {
            if (!m_device.is_cpu()) {
                throw std::runtime_error(
                    "PaiNN float64 diagnostic mode is certified on CPU only; use device=cpu.");
            }
            m_dtype = torch::kFloat64;
        } else {
            throw std::invalid_argument(
                "Unsupported PaiNN precision '" + precision_str +
                "' (expected float32 or float64)");
        }

        // Convert both parameters and floating buffers (including the independent
        // PaiNN and ordered-geometry energy scales)
        // before moving the model to its execution device.  The float64 mode is
        // diagnostic: it promotes the trained FP32 weights and removes FP32
        // roundoff from the forward/autograd evaluation without retraining.
        model->to(m_dtype);
        model->to(m_device);
        std::cout << "[PaiNN] Inference precision: "
                  << (m_dtype == torch::kFloat64 ? "float64" : "float32") << "\n";

        if (m_device.type() == torch::kMPS) {
            constexpr const char* cadence_env =
                "MLCG_MPS_EMPTY_CACHE_EVERY_FORCE_CALLS";
            constexpr std::int64_t default_cadence = 100;
            const bool cadence_overridden = std::getenv(cadence_env) != nullptr;
            m_mps_empty_cache_every_force_calls =
                nonnegative_integer_environment(cadence_env, default_cadence);
            // stderr + endl is intentional: pypresso may not flush C++ stdout
            // when it finalizes MPI.  Every MPS run must attest the effective
            // allocator policy, including the production default.
            std::cerr << "[PaiNN] MPS diagnostic emptyCache cadence: "
                      << m_mps_empty_cache_every_force_calls
                      << " successful force calls ("
                      << (cadence_overridden ? "environment override" : "MPS default")
                      << ")" << std::endl;
        }

        // Report the gauge that belongs to the active learned branch.  The
        // CGnet-exact head-only model deliberately has no embedding/readout,
        // so calling isolated_species_reference_table() there would access an
        // empty ModuleHolder during construction.
        if (model->has_painn_branch()) {
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
        } else {
            std::cout << "[PaiNN] Energy gauge: ordered_geometry_zero_feature_v1 "
                      << "(CGnet-exact head only; no isolated-species table)\n";
        }
        
        std::cout << "[PaiNN] Modello C++ inizializzato e pesi caricati da: " << model_path << "\n";
    } catch (const c10::Error& e) {
        std::cerr << "[PaiNN] Errore nel caricamento del modello: " << e.what() << "\n";
        throw;
    }
}

void PaiNN_ML_Potential::configure_profiling(bool enabled, std::int64_t warmup_calls) {
    if (warmup_calls < 0) {
        throw std::invalid_argument("PaiNN profiling warmup_calls must be non-negative");
    }
    reset_profiling();
    m_profile.enabled = enabled;
    m_profile.warmup_calls = warmup_calls;
}

void PaiNN_ML_Potential::reset_profiling() {
    const bool enabled = m_profile.enabled;
    const std::int64_t warmup_calls = m_profile.warmup_calls;
    m_profile = ProfileAccumulator{};
    m_profile.enabled = enabled;
    m_profile.warmup_calls = warmup_calls;
}

std::string PaiNN_ML_Potential::get_profile_json() const {
    const double calls = static_cast<double>(m_profile.measured_calls);
    const auto mean = [calls](double total) { return calls > 0.0 ? total / calls : 0.0; };
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{";
    out << "\"schema_version\":1,";
    out << "\"enabled\":" << (m_profile.enabled ? "true" : "false") << ",";
    out << "\"warmup_calls\":" << m_profile.warmup_calls << ",";
    out << "\"total_calls\":" << m_profile.total_calls << ",";
    out << "\"measured_calls\":" << m_profile.measured_calls << ",";
    out << "\"timings_ms\":{";
    out << "\"total_mean\":" << mean(m_profile.total_ms) << ",";
    out << "\"node_index_mean\":" << mean(m_profile.node_index_ms) << ",";
    out << "\"neighbor_traversal_mean\":" << mean(m_profile.neighbor_traversal_ms) << ",";
    out << "\"edge_pack_mean\":" << mean(m_profile.edge_pack_ms) << ",";
    out << "\"tensor_inputs_mean\":" << mean(m_profile.tensor_inputs_ms) << ",";
    out << "\"forward_mean\":" << mean(m_profile.forward_ms) << ",";
    out << "\"energy_scalar_mean\":" << mean(m_profile.energy_scalar_ms) << ",";
    out << "\"autograd_mean\":" << mean(m_profile.autograd_ms) << ",";
    out << "\"force_to_cpu_mean\":" << mean(m_profile.force_to_cpu_ms) << ",";
    out << "\"force_scatter_mean\":" << mean(m_profile.force_scatter_ms) << ",";
    const double accounted_ms =
        m_profile.node_index_ms + m_profile.neighbor_traversal_ms +
        m_profile.edge_pack_ms + m_profile.tensor_inputs_ms + m_profile.forward_ms +
        m_profile.energy_scalar_ms + m_profile.autograd_ms +
        m_profile.force_to_cpu_ms + m_profile.force_scatter_ms;
    out << "\"unattributed_cleanup_mean\":"
        << mean(std::max(0.0, m_profile.total_ms - accounted_ms));
    out << "},";
    out << "\"graph\":{";
    out << "\"particles_mean\":" << mean(m_profile.particles_sum) << ",";
    out << "\"particles_max\":" << m_profile.particles_max << ",";
    out << "\"directed_edges_mean\":" << mean(m_profile.directed_edges_sum) << ",";
    out << "\"directed_edges_max\":" << m_profile.directed_edges_max << ",";
    out << "\"physical_pairs_mean\":" << mean(m_profile.physical_pairs_sum) << ",";
    out << "\"physical_pairs_max\":" << m_profile.physical_pairs_max;
    out << "},";
    out << "\"allocation_churn_indicators\":{";
    out << "\"host_payload_lower_bound_bytes_mean\":"
        << mean(m_profile.host_payload_lower_bound_bytes_sum) << ",";
    out << "\"temporary_cpp_containers_per_call\":9,";
    out << "\"note\":\"payload excludes allocator/map-node overhead and libtorch internal allocations\"";
    out << "}";
    out << "}";
    return out.str();
}

void PaiNN_ML_Potential::calculate_forces(
    CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion) {
    calculate_forces_impl(cell_structure, verlet_criterion);

#ifdef __APPLE__
    // calculate_forces_impl has returned, so every per-call tensor and the
    // autograd graph have already been destroyed.  emptyCache can therefore
    // release only unused allocator blocks; it cannot invalidate live tensors.
    if (m_device.type() == torch::kMPS &&
        m_mps_empty_cache_every_force_calls > 0) {
        ++m_successful_force_calls;
        if (m_successful_force_calls % m_mps_empty_cache_every_force_calls == 0) {
            at::mps::getIMPSAllocator()->emptyCache();
        }
    }
#endif
}

void PaiNN_ML_Potential::calculate_forces_impl(
    CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion) {
    // Never expose an energy value from a previous integration step.
    m_last_energy = 0.0;

    const bool profile_enabled = m_profile.enabled;
    const bool profile_this_call =
        profile_enabled && m_profile.total_calls >= m_profile.warmup_calls;
    if (profile_enabled) {
        ++m_profile.total_calls;
    }
    ProfileClock::time_point total_start{};
    ProfileClock::time_point stage_start{};
    if (profile_this_call) {
        total_start = ProfileClock::now();
        stage_start = total_start;
    }
    struct ProfileTotalGuard {
        bool active;
        ProfileClock::time_point start;
        double* total_ms;
        std::int64_t* measured_calls;
        ~ProfileTotalGuard() {
            if (active) {
                *total_ms += elapsed_ms(start, ProfileClock::now());
                ++(*measured_calls);
            }
        }
    } total_guard{
        profile_this_call, total_start, &m_profile.total_ms, &m_profile.measured_calls};

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
    if (profile_this_call) {
        const auto now = ProfileClock::now();
        m_profile.node_index_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }
    if (num_particles == 0) {
        return;
    }

    using PairKey = std::pair<int, int>;
    using Displacement = std::array<double, 3>;
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
                static_cast<double>(d.vec21[0]),
                static_cast<double>(d.vec21[1]),
                static_cast<double>(d.vec21[2])};
        } else {
            r_low_minus_high = {
                static_cast<double>(-d.vec21[0]),
                static_cast<double>(-d.vec21[1]),
                static_cast<double>(-d.vec21[2])};
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
    if (profile_this_call) {
        const auto now = ProfileClock::now();
        m_profile.neighbor_traversal_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }

    std::vector<int64_t> edge_rows;
    std::vector<int64_t> edge_cols;
    std::vector<double> r_ij_data;
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
    if (profile_this_call) {
        const auto now = ProfileClock::now();
        m_profile.edge_pack_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }

    torch::Tensor t_atomic_numbers =
        torch::tensor(atomic_numbers, torch::TensorOptions().dtype(torch::kInt64))
            .to(m_device);

    torch::Tensor t_edge_index;
    torch::Tensor t_r_ij;
    std::vector<int64_t> flat_edges;
    if (num_edges == 0) {
        t_edge_index = torch::empty(
            {2, 0}, torch::TensorOptions().dtype(torch::kInt64).device(m_device));
        t_r_ij = torch::empty(
            {0, 3}, torch::TensorOptions().dtype(m_dtype).device(m_device));
        if (profile_this_call) {
            const auto now = ProfileClock::now();
            m_profile.tensor_inputs_ms += elapsed_ms(stage_start, now);
            stage_start = now;
        }

        // The isolated-species gauge makes this energy exactly zero while
        // retaining a complete forward path and exactly zero forces.
        const torch::Tensor atom_energies =
            model->forward_atom_energies(t_atomic_numbers, t_r_ij, t_edge_index)
                .squeeze(-1);
        if (profile_this_call) {
            const auto now = ProfileClock::now();
            m_profile.forward_ms += elapsed_ms(stage_start, now);
            stage_start = now;
        }
        m_last_energy = sum_atom_energies_for_hamiltonian(atom_energies).item<double>();
        if (profile_this_call) {
            const auto now = ProfileClock::now();
            m_profile.energy_scalar_ms += elapsed_ms(stage_start, now);
            m_profile.particles_sum += static_cast<double>(num_particles);
            m_profile.particles_max = std::max<std::int64_t>(m_profile.particles_max, num_particles);
            const double host_payload =
                static_cast<double>(atomic_numbers.size() * sizeof(int64_t)) +
                static_cast<double>((idx_to_particle.size() + local_ml_particles.size()) * sizeof(Particle*)) +
                static_cast<double>(pid_to_idx.size() * sizeof(std::pair<const int, int>));
            m_profile.host_payload_lower_bound_bytes_sum += host_payload;
        }
        return;
    }

    flat_edges.reserve(static_cast<std::size_t>(2 * num_edges));
    flat_edges.insert(flat_edges.end(), edge_rows.begin(), edge_rows.end());
    flat_edges.insert(flat_edges.end(), edge_cols.begin(), edge_cols.end());
    t_edge_index =
        torch::tensor(flat_edges, torch::TensorOptions().dtype(torch::kInt64))
            .reshape({2, num_edges})
            .to(m_device);

    t_r_ij =
        torch::tensor(r_ij_data, torch::TensorOptions().dtype(m_dtype))
            .reshape({num_edges, 3})
            .to(m_device);
    t_r_ij.set_requires_grad(true);
    if (profile_this_call) {
        const auto now = ProfileClock::now();
        m_profile.tensor_inputs_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }

    const torch::Tensor atom_energies =
        model->forward_atom_energies(t_atomic_numbers, t_r_ij, t_edge_index)
            .squeeze(-1);
    if (profile_this_call) {
        const auto now = ProfileClock::now();
        m_profile.forward_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }
    const torch::Tensor total_energy = sum_atom_energies_for_hamiltonian(atom_energies);

    // Energy and forces are derived from exactly the same scalar Hamiltonian.
    // There are no ghost atom-energy terms in this single-rank graph.
    m_last_energy = total_energy.item<double>();
    if (profile_this_call) {
        const auto now = ProfileClock::now();
        m_profile.energy_scalar_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }
    auto grads = torch::autograd::grad(
        {total_energy}, {t_r_ij}, {torch::ones_like(total_energy)}, false, false);
    if (profile_this_call) {
        const auto now = ProfileClock::now();
        m_profile.autograd_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }
    // Convert only after autograd has finished.  In float64 mode the full
    // forward and force derivative therefore remain FP64; in float32 mode
    // this is merely an exact promotion of the already-computed FP32 force.
    const torch::Tensor f_r_ij = -grads[0].to(torch::kCPU).to(torch::kFloat64);
    auto f_r_ij_acc = f_r_ij.accessor<double, 2>();
    if (profile_this_call) {
        const auto now = ProfileClock::now();
        m_profile.force_to_cpu_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }

    for (int e = 0; e < num_edges; ++e) {
        const int row = static_cast<int>(edge_rows[e]);
        const int col = static_cast<int>(edge_cols[e]);
        const double fx = f_r_ij_acc[e][0];
        const double fy = f_r_ij_acc[e][1];
        const double fz = f_r_ij_acc[e][2];

        idx_to_particle[row]->force()[0] += fx;
        idx_to_particle[row]->force()[1] += fy;
        idx_to_particle[row]->force()[2] += fz;

        idx_to_particle[col]->force()[0] -= fx;
        idx_to_particle[col]->force()[1] -= fy;
        idx_to_particle[col]->force()[2] -= fz;
    }

    if (profile_this_call) {
        const auto now = ProfileClock::now();
        m_profile.force_scatter_ms += elapsed_ms(stage_start, now);
        m_profile.particles_sum += static_cast<double>(num_particles);
        m_profile.directed_edges_sum += static_cast<double>(num_edges);
        m_profile.physical_pairs_sum += static_cast<double>(physical_pairs.size());
        m_profile.particles_max = std::max<std::int64_t>(m_profile.particles_max, num_particles);
        m_profile.directed_edges_max = std::max<std::int64_t>(m_profile.directed_edges_max, num_edges);
        m_profile.physical_pairs_max = std::max<std::int64_t>(
            m_profile.physical_pairs_max, static_cast<std::int64_t>(physical_pairs.size()));
        const double host_payload =
            static_cast<double>(atomic_numbers.size() * sizeof(int64_t)) +
            static_cast<double>((idx_to_particle.size() + local_ml_particles.size()) * sizeof(Particle*)) +
            static_cast<double>(pid_to_idx.size() * sizeof(std::pair<const int, int>)) +
            static_cast<double>(physical_pairs.size() * (sizeof(PairKey) + sizeof(Displacement))) +
            static_cast<double>((edge_rows.size() + edge_cols.size() + flat_edges.size()) * sizeof(int64_t)) +
            static_cast<double>(r_ij_data.size() * sizeof(double));
        m_profile.host_payload_lower_bound_bytes_sum += host_payload;
    }
}
