#!/usr/bin/env python3
"""Generate a benchmark-only PaiNN_ML_Potential.cpp from the production source.

The generated source preserves the production path verbatim when
MLCG_SYNTHETIC_PAINN_CASE_DIR is unset. When enabled, it evaluates the exact
synthetic TorchMD benchmark Hamiltonian on a fixed directed graph. The tracked
production plugin is never modified by this generator.
"""
from __future__ import annotations

import argparse
from pathlib import Path


HELPERS = r'''
// BEGIN MLCG_SYNTHETIC_PAINN_BENCHMARK_OVERRIDE
struct SyntheticBenchmarkCase {
    int particles = 0;
    int num_species = 0;
    int hidden_channels = 0;
    int num_layers = 0;
    int num_rbf = 0;
    double cutoff_nm = 0.0;
    double cutoff_A = 0.0;
    double toxvaerd_alpha = 0.0;
    double energy_scale_kcal = 0.0;
    std::array<double, 3> espresso_translation_A{{0.0, 0.0, 0.0}};
    std::vector<int64_t> edge_rows;
    std::vector<int64_t> edge_cols;
    std::vector<std::array<double, 3>> equilibrium_A;
    std::vector<double> stiffness_kcal_mol_A2;
};

std::optional<SyntheticBenchmarkCase> g_synthetic_benchmark_case;

std::string benchmark_case_dir_from_env() {
    auto const *raw = std::getenv("MLCG_SYNTHETIC_PAINN_CASE_DIR");
    if (raw == nullptr || raw[0] == '\0') return {};
    return std::string(raw);
}

void expect_token(std::istream &input, std::string const &expected, std::string const &path) {
    std::string token;
    if (!(input >> token) || token != expected) {
        throw std::runtime_error(
            "Synthetic PaiNN benchmark parse error in " + path +
            ": expected token '" + expected + "'");
    }
}

SyntheticBenchmarkCase load_synthetic_benchmark_case(
    std::string const &case_dir,
    int num_species,
    int hidden_channels,
    int num_layers,
    int num_rbf,
    double cutoff,
    double toxvaerd_alpha) {
    SyntheticBenchmarkCase data;
    auto const config_path = case_dir + "/config.txt";
    std::ifstream config(config_path);
    if (!config) throw std::runtime_error("Cannot open synthetic PaiNN config: " + config_path);
    expect_token(config, "MLCG_SYNTHETIC_PAINN_CASE_V3", config_path);
    std::string key;
    config >> key >> data.particles;
    if (key != "particles") throw std::runtime_error("Invalid particles field in " + config_path);
    config >> key >> data.num_species;
    if (key != "num_species") throw std::runtime_error("Invalid num_species field in " + config_path);
    config >> key >> data.hidden_channels;
    if (key != "hidden_channels") throw std::runtime_error("Invalid hidden_channels field in " + config_path);
    config >> key >> data.num_layers;
    if (key != "num_layers") throw std::runtime_error("Invalid num_layers field in " + config_path);
    config >> key >> data.num_rbf;
    if (key != "num_rbf") throw std::runtime_error("Invalid num_rbf field in " + config_path);
    config >> key >> data.cutoff_nm;
    if (key != "cutoff_nm") throw std::runtime_error("Invalid cutoff_nm field in " + config_path);
    config >> key >> data.cutoff_A;
    if (key != "cutoff_A") throw std::runtime_error("Invalid cutoff_A field in " + config_path);
    config >> key >> data.toxvaerd_alpha;
    if (key != "toxvaerd_alpha") throw std::runtime_error("Invalid toxvaerd_alpha field in " + config_path);
    config >> key >> data.energy_scale_kcal;
    if (key != "energy_scale_kcal") throw std::runtime_error("Invalid energy_scale_kcal field in " + config_path);
    config >> key >> data.espresso_translation_A[0] >> data.espresso_translation_A[1] >> data.espresso_translation_A[2];
    if (key != "espresso_translation_A") throw std::runtime_error("Invalid espresso_translation_A field in " + config_path);
    if (!config) throw std::runtime_error("Truncated synthetic PaiNN config: " + config_path);

    auto close = [](double a, double b) {
        return std::abs(a - b) <= 1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
    };
    if (data.num_species != num_species || data.hidden_channels != hidden_channels ||
        data.num_layers != num_layers || data.num_rbf != num_rbf ||
        !close(data.cutoff_nm, cutoff) || !close(data.toxvaerd_alpha, toxvaerd_alpha) ||
        !close(data.cutoff_A, 10.0 * data.cutoff_nm)) {
        throw std::runtime_error(
            "Synthetic PaiNN case/config mismatch with activate_painn_potential arguments");
    }

    auto const graph_path = case_dir + "/graph.txt";
    std::ifstream graph(graph_path);
    if (!graph) throw std::runtime_error("Cannot open fixed graph: " + graph_path);
    std::size_t edges = 0;
    graph >> edges;
    data.edge_rows.reserve(edges);
    data.edge_cols.reserve(edges);
    for (std::size_t e = 0; e < edges; ++e) {
        int64_t row = -1, col = -1;
        graph >> row >> col;
        if (!graph || row < 0 || col < 0 || row >= data.particles || col >= data.particles || row == col) {
            throw std::runtime_error("Invalid fixed edge in " + graph_path);
        }
        data.edge_rows.push_back(row);
        data.edge_cols.push_back(col);
    }

    auto const harmonic_path = case_dir + "/harmonic_torchmd_units.txt";
    std::ifstream harmonic(harmonic_path);
    if (!harmonic) throw std::runtime_error("Cannot open harmonic data: " + harmonic_path);
    int n_harmonic = 0;
    harmonic >> n_harmonic;
    if (n_harmonic != data.particles) {
        throw std::runtime_error("Harmonic particle count mismatch in " + harmonic_path);
    }
    data.equilibrium_A.resize(data.particles);
    data.stiffness_kcal_mol_A2.resize(data.particles);
    for (int i = 0; i < data.particles; ++i) {
        auto &r0 = data.equilibrium_A[static_cast<std::size_t>(i)];
        double k = 0.0;
        harmonic >> r0[0] >> r0[1] >> r0[2] >> k;
        if (!harmonic || !(k > 0.0)) {
            throw std::runtime_error("Invalid harmonic row in " + harmonic_path);
        }
        data.stiffness_kcal_mol_A2[static_cast<std::size_t>(i)] = k;
    }
    return data;
}

void load_synthetic_benchmark_weights(
    PaiNNModel &model, std::string const &case_dir, double energy_scale_kcal) {
    auto const path = case_dir + "/weights.txt";
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open synthetic PaiNN weights: " + path);
    expect_token(input, "MLCG_SYNTHETIC_PAINN_WEIGHTS_V1", path);
    std::size_t tensor_count = 0;
    input >> tensor_count;
    std::unordered_map<std::string, std::vector<double>> weights;
    for (std::size_t i = 0; i < tensor_count; ++i) {
        std::string name;
        std::size_t numel = 0;
        input >> name >> numel;
        if (!input || name.empty() || numel == 0 || weights.count(name) != 0) {
            throw std::runtime_error("Invalid tensor header in " + path);
        }
        auto &values = weights[name];
        values.resize(numel);
        for (std::size_t j = 0; j < numel; ++j) {
            input >> values[j];
            if (!input || !std::isfinite(values[j])) {
                throw std::runtime_error("Invalid tensor payload for " + name + " in " + path);
            }
        }
    }
    expect_token(input, "END", path);

    std::size_t copied = 0;
    torch::NoGradGuard no_grad;
    for (auto const &item : model->named_parameters(true)) {
        auto const it = weights.find(item.key());
        if (it == weights.end()) {
            throw std::runtime_error("Missing synthetic PaiNN parameter: " + item.key());
        }
        auto param = item.value();
        if (static_cast<std::size_t>(param.numel()) != it->second.size()) {
            throw std::runtime_error("Synthetic PaiNN parameter size mismatch: " + item.key());
        }
        auto source = torch::from_blob(
            const_cast<double *>(it->second.data()), param.sizes(),
            torch::TensorOptions().dtype(torch::kFloat64)).clone();
        param.copy_(source.to(param.options()));
        ++copied;
    }
    if (copied != weights.size()) {
        throw std::runtime_error(
            "Synthetic PaiNN weights contain unexpected parameters: file=" +
            std::to_string(weights.size()) + " model=" + std::to_string(copied));
    }
    // Python SyntheticPaiNN applies residual_scale after the atom-energy sum.
    // Keep the C++ per-atom gauge unscaled, then apply the shared scalar in
    // calculate_synthetic_benchmark_forces in the same operation order.
    (void)energy_scale_kcal;
    model->energy_scale.fill_(1.0);
}

double calculate_synthetic_benchmark_forces(
    PaiNNModel &model,
    torch::Device const &device,
    torch::Dtype dtype,
    CellStructure &cell_structure,
    SyntheticBenchmarkCase const &bench) {
    constexpr double NM_TO_ANGSTROM = 10.0;
    constexpr double KCAL_TO_KJ = 4.184;
    constexpr double FORCE_KCAL_A_TO_KJ_NM = 41.84;

    std::vector<Particle *> particles;
    for (auto &p : cell_structure.local_particles()) {
        if (p.type() < bench.num_species) particles.push_back(&p);
    }
    std::sort(
        particles.begin(), particles.end(),
        [](Particle const *a, Particle const *b) { return a->id() < b->id(); });
    if (static_cast<int>(particles.size()) != bench.particles) {
        throw std::runtime_error(
            "Synthetic PaiNN benchmark expected " + std::to_string(bench.particles) +
            " ML particles, found " + std::to_string(particles.size()));
    }

    std::vector<int64_t> atomic_numbers;
    std::vector<double> positions_A;
    atomic_numbers.reserve(particles.size());
    positions_A.reserve(3 * particles.size());
    for (auto const *p : particles) {
        atomic_numbers.push_back(p->type());
        auto const &pos = p->pos();
        // ESPResSo stores a rigidly translated copy of the state so no initial
        // coordinate is folded across a periodic boundary. Undo that translation
        // in double precision before constructing the Torch tensor, so LibTorch
        // sees the original TorchMD numerical coordinates even in float32 mode.
        positions_A.push_back(static_cast<double>(pos[0]) * NM_TO_ANGSTROM - bench.espresso_translation_A[0]);
        positions_A.push_back(static_cast<double>(pos[1]) * NM_TO_ANGSTROM - bench.espresso_translation_A[1]);
        positions_A.push_back(static_cast<double>(pos[2]) * NM_TO_ANGSTROM - bench.espresso_translation_A[2]);
    }

    auto const num_edges = static_cast<int64_t>(bench.edge_rows.size());
    std::vector<int64_t> flat_edges;
    flat_edges.reserve(static_cast<std::size_t>(2 * num_edges));
    flat_edges.insert(flat_edges.end(), bench.edge_rows.begin(), bench.edge_rows.end());
    flat_edges.insert(flat_edges.end(), bench.edge_cols.begin(), bench.edge_cols.end());

    std::vector<double> equilibrium_A;
    equilibrium_A.reserve(3 * particles.size());
    for (auto const &xyz : bench.equilibrium_A) {
        equilibrium_A.push_back(xyz[0]);
        equilibrium_A.push_back(xyz[1]);
        equilibrium_A.push_back(xyz[2]);
    }

    auto t_atomic_numbers =
        torch::tensor(atomic_numbers, torch::TensorOptions().dtype(torch::kInt64)).to(device);
    auto t_edge_index =
        torch::tensor(flat_edges, torch::TensorOptions().dtype(torch::kInt64))
            .reshape({2, num_edges}).to(device);
    auto t_positions =
        torch::tensor(positions_A, torch::TensorOptions().dtype(dtype))
            .reshape({1, bench.particles, 3}).to(device);
    t_positions.set_requires_grad(true);
    auto t_equilibrium =
        torch::tensor(equilibrium_A, torch::TensorOptions().dtype(dtype))
            .reshape({1, bench.particles, 3}).to(device);
    auto t_stiffness =
        torch::tensor(bench.stiffness_kcal_mol_A2, torch::TensorOptions().dtype(dtype))
            .reshape({1, bench.particles, 1}).to(device);

    auto xyz = t_positions.index({0});
    auto row = t_edge_index.index({0});
    auto col = t_edge_index.index({1});
    auto r_ij = xyz.index_select(0, row) - xyz.index_select(0, col);
    auto atom_energies = model->forward_atom_energies(t_atomic_numbers, r_ij, t_edge_index)
                             .squeeze(-1);
    // Native-dtype reduction deliberately matches the Python synthetic benchmark.
    auto ml_energy_kcal = atom_energies.sum() * bench.energy_scale_kcal;
    auto delta = t_positions - t_equilibrium;
    auto harmonic_energy_kcal = 0.5 * (t_stiffness * delta * delta).sum();
    auto total_energy_kcal = ml_energy_kcal + harmonic_energy_kcal;

    auto grads = torch::autograd::grad(
        {total_energy_kcal}, {t_positions}, {torch::ones_like(total_energy_kcal)},
        false, false);
    auto force_kj_nm = (-grads[0] * FORCE_KCAL_A_TO_KJ_NM)
                           .to(torch::kCPU).to(torch::kFloat64).contiguous();
    auto force_acc = force_kj_nm.accessor<double, 3>();
    for (int i = 0; i < bench.particles; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            particles[static_cast<std::size_t>(i)]->force()[axis] += force_acc[0][i][axis];
        }
    }
    return total_energy_kcal.item<double>() * KCAL_TO_KJ;
}
// END MLCG_SYNTHETIC_PAINN_BENCHMARK_OVERRIDE
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    text = source.read_text(encoding="utf-8")
    if "MLCG_SYNTHETIC_PAINN_BENCHMARK_OVERRIDE" in text:
        raise RuntimeError("source already contains the synthetic benchmark override")

    include_anchor = '#include <iomanip>\n#include <iostream>\n'
    text = replace_once(
        text,
        include_anchor,
        '#include <iomanip>\n#include <iostream>\n#include <cstdlib>\n#include <fstream>\n#include <optional>\n',
        "include",
    )
    namespace_anchor = '\n} // namespace\n\nPaiNN_ML_Potential::PaiNN_ML_Potential'
    text = replace_once(
        text,
        namespace_anchor,
        '\n' + HELPERS + '\n} // namespace\n\nPaiNN_ML_Potential::PaiNN_ML_Potential',
        "anonymous namespace",
    )

    load_anchor = '''    try {\n        torch::load(model, model_path);\n        model->eval(); // Mette il modello in modalità inferenza\n'''
    load_replacement = '''    try {\n        auto const benchmark_case_dir = benchmark_case_dir_from_env();\n        if (!benchmark_case_dir.empty()) {\n            // Canonical parameters are exported in float64; round only once when\n            // the requested benchmark precision is selected below.\n            model->to(torch::kFloat64);\n            g_synthetic_benchmark_case = load_synthetic_benchmark_case(\n                benchmark_case_dir, num_species, hidden_channels, n_layers, num_rbf,\n                cutoff, toxvaerd_alpha);\n            load_synthetic_benchmark_weights(\n                model, benchmark_case_dir, g_synthetic_benchmark_case->energy_scale_kcal);\n            // The benchmark tensors are evaluated numerically in the same units\n            // as TorchMD (Angstrom, kcal/mol). m_cutoff remains nm for ESPResSo.\n            model->cutoff_radius = g_synthetic_benchmark_case->cutoff_A;\n            std::cout << "[PaiNN benchmark] Loaded exact fixed-graph synthetic case from: "\n                      << benchmark_case_dir << "\\n";\n        } else {\n            g_synthetic_benchmark_case.reset();\n            torch::load(model, model_path);\n        }\n        model->eval(); // Mette il modello in modalità inferenza\n'''
    text = replace_once(text, load_anchor, load_replacement, "model-load")

    force_anchor = '''void PaiNN_ML_Potential::calculate_forces(CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion) {\n    // Never expose an energy value from a previous integration step.\n    m_last_energy = 0.0;\n'''
    force_replacement = force_anchor + '''\n    if (g_synthetic_benchmark_case.has_value()) {\n        m_last_energy = calculate_synthetic_benchmark_forces(\n            model, m_device, m_dtype, cell_structure, *g_synthetic_benchmark_case);\n        return;\n    }\n'''
    text = replace_once(text, force_anchor, force_replacement, "calculate_forces")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"[PASS] generated benchmark-only ESPResSo PaiNN override: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
