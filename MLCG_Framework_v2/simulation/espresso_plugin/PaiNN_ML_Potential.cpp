#include "PaiNN_ML_Potential.hpp"
#include "BoxGeometry.hpp"
#include "Particle.hpp"
#include "cells.hpp"
#include "exclusions.hpp"
#include "system/System.hpp"

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
#include <unordered_set>
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

// Profiling only: CUDA kernels are asynchronous, so a stage boundary must wait
// for the device or the time of one stage is booked to the next one that
// synchronises (.item(), .to(kCPU)).  Never called when profiling is off.
void profile_device_sync(torch::Device const &device) {
    if (device.is_cuda()) {
        torch::cuda::synchronize();
    }
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
    int ordered_geometry_copies,
    bool tel22_shared_geometry,
    const std::string& device_str,
    const std::string& precision_str)
    : m_cutoff(cutoff), m_num_species(num_species),
      m_tel22_shared_geometry(tel22_shared_geometry) {
    
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
        ordered_geometry_head_only,
        ordered_geometry_copies,
        tel22_shared_geometry);
    
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
        // In MD le forze sono -grad E via autograd: con le matmul in TF32 anche
        // il passo all'indietro verrebbe arrotondato, e la forza non sarebbe
        // piu' il gradiente esatto dell'energia calcolata -- un'incoerenza che
        // rompe la conservazione dell'energia in NVE.  LibTorch usa gia' FP32
        // pieno di default, ma la variabile d'ambiente
        // TORCH_ALLOW_TF32_CUBLAS_OVERRIDE lo cambia per tutto il processo:
        // qui lo si fissa esplicitamente, cosi' la garanzia non dipende
        // dall'ambiente del job.  Il TF32 resta un'opzione del solo trainer.
        if (m_device.is_cuda()) {
            at::globalContext().setFloat32MatmulPrecision("highest");
        }
        {
            const char* graph_env = std::getenv("MLCG_PAINN_GRAPH");
            const std::string graph_mode = graph_env ? std::string(graph_env) : "legacy";
            if (graph_mode == "device") {
                m_device_graph = true;
            } else if (graph_mode != "legacy") {
                throw std::invalid_argument(
                    "MLCG_PAINN_GRAPH must be 'legacy' or 'device', got '" + graph_mode + "'");
            }
            // stderr + endl: always visible even if pypresso does not flush stdout.
            std::cerr << "[PaiNN] Graph path: "
                      << (m_device_graph
                              ? "device (pair search + node forces on the model device)"
                              : "legacy (ESPResSo pair loop + host graph)")
                      << std::endl;
        }
        std::cout << "[PaiNN] Inference precision: "
                  << (m_dtype == torch::kFloat64 ? "float64" : "float32")
                  << (m_device.is_cuda() ? " (TF32 disabilitato)" : "") << "\n";

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
    out << "\"graph_path\":\"" << (m_device_graph ? "device" : "legacy") << "\",";
    out << "\"model_device\":\"" << m_device.str() << "\",";
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

double PaiNN_ML_Potential::get_last_energy() const {
    if (m_energy_pending) {
        m_last_energy = m_energy_tensor.to(torch::kCPU).to(torch::kFloat64).item<double>();
        m_energy_pending = false;
        m_energy_tensor = torch::Tensor();
    }
    return m_last_energy;
}

void PaiNN_ML_Potential::calculate_forces(
    CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion) {
    if (m_device_graph) {
        calculate_forces_device_graph(cell_structure);
    } else {
        calculate_forces_impl(cell_structure, verlet_criterion);
    }

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

void PaiNN_ML_Potential::validate_tel22_layout(
    std::vector<Particle*> const& particles,
    std::vector<int64_t> const& atomic_numbers) {
    if (static_cast<int>(particles.size()) != TEL22_SHARED_COPIES * TEL22_SHARED_SITES_PER_COPY) {
        throw std::runtime_error(
            "TEL22 shared head requires exactly 820 ordered physical sites");
    }
    int node = 0;
    for (int copy = 0; copy < TEL22_SHARED_COPIES; ++copy) {
        for (int residue = 0; residue < 22; ++residue) {
            const bool adenine =
                residue == 0 || residue == 6 || residue == 12 || residue == 18;
            const bool thymine =
                residue == 4 || residue == 5 || residue == 10 || residue == 11 ||
                residue == 16 || residue == 17;
            const int site_count = (adenine || thymine) ? 1 : 6;
            for (int site = 0; site < site_count; ++site, ++node) {
                const int expected_type = site_count == 1
                    ? (adenine ? 0 : 1)
                    : 2 + site;
                const int expected_molecule = copy * 22 + residue;
                if (atomic_numbers.at(node) != expected_type ||
                    particles.at(node)->mol_id() != expected_molecule) {
                    throw std::runtime_error(
                        "TEL22 shared head particle order/type/molecule contract mismatch");
                }
            }
        }
    }
    m_tel22_layout_validated = true;
    std::cout << "[PaiNN] TEL22 10x82 shared-head particle contract validated.\n";
}

void PaiNN_ML_Potential::calculate_forces_impl(
    CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion) {
    // Never expose an energy value from a previous integration step.
    m_last_energy = 0.0;
    m_energy_pending = false;
    m_energy_tensor = torch::Tensor();

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
    if (m_tel22_shared_geometry && !m_tel22_layout_validated) {
        validate_tel22_layout(idx_to_particle, atomic_numbers);
    }
    if (profile_this_call) {
        profile_device_sync(m_device);
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
        profile_device_sync(m_device);
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
        profile_device_sync(m_device);
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
        profile_device_sync(m_device);
        const auto now = ProfileClock::now();
        m_profile.tensor_inputs_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }

    const torch::Tensor atom_energies =
        model->forward_atom_energies(t_atomic_numbers, t_r_ij, t_edge_index)
            .squeeze(-1);
    if (profile_this_call) {
        profile_device_sync(m_device);
        const auto now = ProfileClock::now();
        m_profile.forward_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }
    const torch::Tensor total_energy = sum_atom_energies_for_hamiltonian(atom_energies);

    // Energy and forces are derived from exactly the same scalar Hamiltonian.
    // There are no ghost atom-energy terms in this single-rank graph.
    m_last_energy = total_energy.item<double>();
    if (profile_this_call) {
        profile_device_sync(m_device);
        const auto now = ProfileClock::now();
        m_profile.energy_scalar_ms += elapsed_ms(stage_start, now);
        stage_start = now;
    }
    auto grads = torch::autograd::grad(
        {total_energy}, {t_r_ij}, {torch::ones_like(total_energy)}, false, false);
    if (profile_this_call) {
        profile_device_sync(m_device);
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
        profile_device_sync(m_device);
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
        profile_device_sync(m_device);
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

// ---------------------------------------------------------------------------
// Device-resident graph path (MLCG_PAINN_GRAPH=device).
//
// Same Hamiltonian as the legacy path:
//   * nodes   = local ML particles (type < num_species), ordered by particle id;
//   * edges   = every intermolecular pair with |r_ij| <= cutoff, both
//               directions, r_ij = x_row - x_col under the minimum image;
//   * E       = sum of atom energies (float64 accumulator on CPU);
//   * F       = -dE/dx.
// What changes is where the graph is built: instead of walking ESPResSo's
// pair loop into a std::map and packing host vectors (~14 ms/step for TEL26),
// a static list of candidate pairs (all intermolecular i<j, built once) is
// filtered on the model device.  Positions go host->device as one N x 3
// float64 block, autograd is taken w.r.t. those positions, and N x 3 node
// forces come back.  The minimum image and the r_ij differences are evaluated
// in float64 and only then cast to the model dtype, exactly like the legacy
// path (ESPResSo's double vec21 cast to float32).  The edge order (pairs
// sorted by (low, high), directions interleaved) also matches the legacy one.
//
// Limits: single MPI rank (checked against the ghost layer), cuboid periodic
// or open box (no Lees-Edwards), cutoff < L/2 on periodic axes, and an
// O(N^2) candidate list (fine up to a few 10^4 sites).
// ---------------------------------------------------------------------------

void PaiNN_ML_Potential::rebuild_device_graph_cache(
    CellStructure& cell_structure, std::vector<Particle*> const& particles) {
    const auto n = static_cast<std::int64_t>(particles.size());
    constexpr std::int64_t max_nodes = 20000;
    if (n > max_nodes) {
        throw std::runtime_error(
            "MLCG_PAINN_GRAPH=device uses an O(N^2) candidate pair list; " +
            std::to_string(n) + " ML sites exceed the supported " +
            std::to_string(max_nodes) + ". Use MLCG_PAINN_GRAPH=legacy.");
    }

    std::vector<int64_t> species;
    std::vector<int64_t> molecules;
    species.reserve(particles.size());
    molecules.reserve(particles.size());
    m_graph_ids.clear();
    m_graph_ids.reserve(particles.size());
    for (auto const* p : particles) {
        m_graph_ids.push_back(p->id());
        species.push_back(p->type());
        molecules.push_back(p->mol_id());
    }

    // Single-rank contract: every ML ghost must be a periodic alias of a
    // local ML particle.  On more ranks a ghost may be a remote particle,
    // which this path would silently ignore.
    std::unordered_set<int> local_ids(m_graph_ids.begin(), m_graph_ids.end());
    if (local_ids.size() != m_graph_ids.size()) {
        throw std::runtime_error("PaiNN device graph: duplicate local particle id");
    }
    for (auto const& ghost : cell_structure.ghost_particles()) {
        if (ghost.type() < m_num_species && local_ids.count(ghost.id()) == 0) {
            throw std::runtime_error(
                "PaiNN device graph found an ML ghost that is not a periodic image "
                "of a local particle (multi-rank run?). Run with one MPI rank or "
                "use MLCG_PAINN_GRAPH=legacy.");
        }
    }

    if (m_tel22_shared_geometry && !m_tel22_layout_validated) {
        validate_tel22_layout(particles, species);
    }

    const auto long_cpu = torch::TensorOptions().dtype(torch::kInt64);
    const auto long_dev = long_cpu.device(m_device);
    m_graph_species = torch::tensor(species, long_cpu).to(m_device);
    auto mol = torch::tensor(molecules, long_cpu).to(m_device);
    auto tri = torch::triu_indices(n, n, 1, long_dev);
    auto ti = tri.index({0});
    auto tj = tri.index({1});
    auto keep = mol.index_select(0, ti) != mol.index_select(0, tj);
    m_pair_i = ti.masked_select(keep).contiguous();
    m_pair_j = tj.masked_select(keep).contiguous();

    // Plain (pageable) host memory on purpose.  The buffer is N x 3 doubles
    // (~20 KB for TEL26), so pinning gains nothing measurable, while a pinned
    // block used by a non_blocking copy is released through CUDA's caching
    // host allocator: when the global potential is destroyed at process exit,
    // after the CUDA runtime has started unloading, that release throws inside
    // a destructor and aborts the process (job 59217069).
    m_pos_host = torch::empty({n, 3}, torch::TensorOptions().dtype(torch::kFloat64));

    std::cerr << "[PaiNN] Device graph cache built: N=" << n
              << " candidate intermolecular pairs=" << m_pair_i.size(0)
              << " device=" << m_device.str() << std::endl;
}

void PaiNN_ML_Potential::calculate_forces_device_graph(CellStructure& cell_structure) {
    m_last_energy = 0.0;
    m_energy_pending = false;
    m_energy_tensor = torch::Tensor();

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
    const auto stage = [&](double& accumulator) {
        if (!profile_this_call) {
            return;
        }
        profile_device_sync(m_device);
        const auto now = ProfileClock::now();
        accumulator += elapsed_ms(stage_start, now);
        stage_start = now;
    };

    // --- nodes: local ML particles ordered by id (same layout as legacy) ---
    std::vector<Particle*> particles;
    for (auto& p : cell_structure.local_particles()) {
        if (p.type() < m_num_species) {
            particles.push_back(&p);
        }
    }
    std::sort(
        particles.begin(), particles.end(),
        [](Particle const* lhs, Particle const* rhs) { return lhs->id() < rhs->id(); });
    const auto n = static_cast<std::int64_t>(particles.size());
    if (n == 0) {
        return;
    }
    bool rebuild = static_cast<std::size_t>(n) != m_graph_ids.size();
    if (!rebuild) {
        for (std::int64_t k = 0; k < n; ++k) {
            if (particles[k]->id() != m_graph_ids[k]) {
                rebuild = true;
                break;
            }
        }
    }
    if (rebuild) {
        rebuild_device_graph_cache(cell_structure, particles);
    }
    stage(m_profile.node_index_ms);

    // --- positions host -> device, box ---
    {
        auto* buffer = m_pos_host.data_ptr<double>();
        for (std::int64_t k = 0; k < n; ++k) {
            auto const& pos = particles[k]->pos();
            buffer[3 * k + 0] = pos[0];
            buffer[3 * k + 1] = pos[1];
            buffer[3 * k + 2] = pos[2];
        }
    }
    // copy=true: on CPU the model input must not alias the reused buffer.
    // Synchronous copy from pageable memory: the buffer can be rewritten as
    // soon as .to() returns.
    auto x = m_pos_host.to(m_device, torch::kFloat64, /*non_blocking=*/false, /*copy=*/true);

    {
        auto const& box_geo = *::System::get_system().box_geo;
        auto const& length = box_geo.length();
        bool changed = !m_box_t.defined();
        for (unsigned axis = 0; axis < 3; ++axis) {
            if (length[axis] != m_box_cached[axis] ||
                box_geo.periodic(axis) != m_periodic_cached[axis]) {
                changed = true;
            }
        }
        if (changed) {
            std::vector<double> box(3), inv(3);
            for (unsigned axis = 0; axis < 3; ++axis) {
                const bool periodic = box_geo.periodic(axis);
                if (periodic && !(m_cutoff < 0.5 * length[axis])) {
                    throw std::runtime_error(
                        "PaiNN device graph requires cutoff < L/2 on periodic axes");
                }
                m_box_cached[axis] = length[axis];
                m_periodic_cached[axis] = periodic;
                box[axis] = length[axis];
                inv[axis] = periodic ? 1.0 / length[axis] : 0.0;
            }
            const auto dbl = torch::TensorOptions().dtype(torch::kFloat64);
            m_box_t = torch::tensor(box, dbl).reshape({1, 3}).to(m_device);
            m_box_inv_t = torch::tensor(inv, dbl).reshape({1, 3}).to(m_device);
        }
    }
    stage(m_profile.tensor_inputs_ms);

    // --- pair search on the device (no autograd) ---
    torch::Tensor pair_i, pair_j, shift;
    {
        torch::NoGradGuard no_grad;
        auto dr = x.index_select(0, m_pair_i) - x.index_select(0, m_pair_j);
        auto image_shift = torch::round(dr * m_box_inv_t) * m_box_t;
        auto r = dr - image_shift;
        auto d2 = (r * r).sum(1);
        auto selected = torch::nonzero(d2 <= m_cutoff * m_cutoff).squeeze(1);
        pair_i = m_pair_i.index_select(0, selected);
        pair_j = m_pair_j.index_select(0, selected);
        shift = image_shift.index_select(0, selected);
    }
    const std::int64_t num_pairs = pair_i.size(0);
    stage(m_profile.neighbor_traversal_ms);

    // --- graph tensors (differentiable w.r.t. x) ---
    x.set_requires_grad(true);
    auto r_low_minus_high =
        x.index_select(0, pair_i) - x.index_select(0, pair_j) - shift;
    auto edge_rows = torch::stack({pair_i, pair_j}, 1).reshape({-1});
    auto edge_cols = torch::stack({pair_j, pair_i}, 1).reshape({-1});
    auto edge_index = torch::stack({edge_rows, edge_cols}, 0);
    auto r_ij = torch::stack({r_low_minus_high, -r_low_minus_high}, 1)
                    .reshape({-1, 3})
                    .to(m_dtype);
    stage(m_profile.edge_pack_ms);

    const torch::Tensor atom_energies =
        model->forward_atom_energies(m_graph_species, r_ij, edge_index).squeeze(-1);
    const torch::Tensor total_energy = sum_atom_energies_for_hamiltonian(atom_energies);
    stage(m_profile.forward_ms);

    // Energy stays on the device; get_last_energy() converts it on demand.
    m_energy_tensor = total_energy.detach();
    m_energy_pending = true;

    if (num_pairs == 0) {
        // Isolated-species gauge: zero energy and exactly zero forces.
        if (profile_this_call) {
            m_profile.particles_sum += static_cast<double>(n);
            m_profile.particles_max = std::max<std::int64_t>(m_profile.particles_max, n);
        }
        return;
    }

    auto grads = torch::autograd::grad(
        {total_energy}, {x}, {torch::ones_like(total_energy)}, false, false);
    stage(m_profile.autograd_ms);

    const torch::Tensor node_forces = grads[0].neg().to(torch::kCPU).contiguous();
    auto f = node_forces.accessor<double, 2>();
    stage(m_profile.force_to_cpu_ms);

    for (std::int64_t k = 0; k < n; ++k) {
        auto& force = particles[k]->force();
        force[0] += f[k][0];
        force[1] += f[k][1];
        force[2] += f[k][2];
    }
    stage(m_profile.force_scatter_ms);

    if (profile_this_call) {
        m_profile.particles_sum += static_cast<double>(n);
        m_profile.directed_edges_sum += static_cast<double>(2 * num_pairs);
        m_profile.physical_pairs_sum += static_cast<double>(num_pairs);
        m_profile.particles_max = std::max<std::int64_t>(m_profile.particles_max, n);
        m_profile.directed_edges_max =
            std::max<std::int64_t>(m_profile.directed_edges_max, 2 * num_pairs);
        m_profile.physical_pairs_max =
            std::max<std::int64_t>(m_profile.physical_pairs_max, num_pairs);
        m_profile.host_payload_lower_bound_bytes_sum +=
            static_cast<double>(n * 3 * sizeof(double) * 2);
    }
}
