// =====================================================================
// train_relent: un passo di allenamento a ENTROPIA RELATIVA del residuo PaiNN.
//
// PERCHE'.  Il force matching minimizza <|F_AA - F_theta|^2>: con base completa
// e dati infiniti l'ottimo e' il PMF, ma con una base incompleta l'ottimo sulle
// forze non coincide con quello sulla struttura, e su TEL26 ogni residuo
// allenato cosi' peggiora le g(r) rispetto ai soli prior (lp1: 0.976 intra,
// d64: 0.94).  L'entropia relativa ottimizza invece proprio la distribuzione:
//
//   S_rel(theta) = < ln p_AA(M(x)) / p_theta(M(x)) >_AA
//   dS/dtheta    = beta [ <dU/dtheta>_AA - <dU/dtheta>_theta ]
//
// Serve solo la posizione dei frame (niente forze, niente rumore sulle forze),
// e l'ottimo e' quello in cui le medie di dU/dtheta sulle due distribuzioni
// coincidono -- per una base di coppie, e' l'uguaglianza delle g(r).
//
// SCHEMA ITERATIVO.  Un'invocazione e' UNA iterazione: i campioni CG vengono
// da una simulazione fatta col modello theta_0 (quello passato con --in).
// Durante i passi di gradiente il modello si allontana da theta_0 e la media
// CG si stima per ripesatura,
//
//   w_i ~ exp(-beta [U_theta(x_i) - U_theta0(x_i)]),
//
// finche' la dimensione efficace del campione ESS = 1/sum w_i^2 resta sopra
// --ess-min volte il numero di frame.  Poi si simula di nuovo (06_relent.sh).
// Le priors non dipendono da theta, quindi si cancellano in U_theta - U_theta0
// e in dU/dtheta: qui basta l'energia della rete.
//
// CONTROLLO.  Con la ripesatura si stima anche la variazione esatta di S_rel
// rispetto a theta_0 (nessuna costante ignota):
//
//   Delta S = beta <U_theta - U_theta0>_AA + ln < exp(-beta (U_theta - U_theta0)) >_theta0
//
// Si misura su un sottoinsieme dei frame AA di allenamento e sulla coda della
// traiettoria AA tenuta fuori (--holdout-frac).  Si salva il modello con il
// Delta S di holdout piu' basso: e' l'early stopping sulla grandezza che si
// sta ottimizzando, non su un surrogato.
//
// INIZIALIZZAZIONE A ZERO.  --zero-init azzera l'ultimo strato del readout:
// l'energia della rete e' identicamente nulla e la prima simulazione riproduce
// i soli prior.  E' il punto di partenza giusto quando i prior sono gia' buoni
// sulla struttura: ogni passo puo' solo abbassare S_rel.
//
// Uso:
//   train_relent --config C.json --aa AA.bin --out M1.pt (--in M0.pt | --zero-init)
//                [--cg CG.bin] [--steps 100] [--lr 1e-3] ...
//   train_relent --config C.json --aa AA.bin --cg CG.bin --in M0.pt --check-gradient
// =====================================================================
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "PaiNN_Architecture.hpp"

using json = nlohmann::json;

// ---------------------------------------------------------------------
// Dataset: solo le posizioni.  Stesso formato binario di train_painn e di
// build_cg_dataset.py; forze e coppie si leggono e si scartano.
// ---------------------------------------------------------------------
struct PositionSet {
    std::string path;
    int64_t num_frames = 0;
    int64_t num_sites = 0;
    std::vector<int64_t> site_types;  // [N], uguale per tutti i frame
    std::vector<int64_t> site_mol;    // [N], molecola (corpo) di ogni sito
    std::vector<float> positions;     // [F*N*3]
    std::vector<float> boxes;         // [F*3]
};

static void read_exact(std::ifstream& file, void* dst, std::size_t nbytes,
                       const std::string& what) {
    file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(nbytes));
    if (!file) throw std::runtime_error("dataset troncato leggendo " + what);
}

static PositionSet read_positions(const std::string& path, int num_species) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("impossibile aprire " + path);
    PositionSet set;
    set.path = path;
    int num_frames = 0;
    read_exact(file, &num_frames, sizeof(int), "num_frames");
    if (num_frames <= 0 || num_frames > 10000000) {
        throw std::runtime_error("num_frames non valido in " + path);
    }
    set.num_frames = num_frames;
    for (int f = 0; f < num_frames; ++f) {
        int num_molecules = 0, num_total_sites = 0;
        float box[3];
        read_exact(file, &num_molecules, sizeof(int), "num_molecules");
        read_exact(file, &num_total_sites, sizeof(int), "num_total_sites");
        read_exact(file, box, 3 * sizeof(float), "box");
        for (int k = 0; k < 3; ++k) {
            if (!std::isfinite(box[k]) || box[k] <= 0.0f) {
                throw std::runtime_error("scatola non valida al frame " + std::to_string(f));
            }
            set.boxes.push_back(box[k]);
        }
        if (f == 0) {
            set.num_sites = num_total_sites;
            set.positions.reserve(static_cast<std::size_t>(num_frames) * num_total_sites * 3);
        } else if (num_total_sites != set.num_sites) {
            throw std::runtime_error("numero di siti diverso fra i frame di " + path);
        }
        int64_t site_cursor = 0;
        for (int m = 0; m < num_molecules; ++m) {
            int mol_id = 0, num_sites = 0;
            float skip[9];
            read_exact(file, &mol_id, sizeof(int), "molecule_id");
            read_exact(file, &num_sites, sizeof(int), "num_sites");
            if (mol_id != m) throw std::runtime_error("molecule_id non sequenziale in " + path);
            read_exact(file, skip, 9 * sizeof(float), "centro/forza/coppia");
            for (int s = 0; s < num_sites; ++s) {
                int type = 0;
                float xyz[3];
                read_exact(file, &type, sizeof(int), "site_type");
                read_exact(file, xyz, 3 * sizeof(float), "site xyz");
                if (type < 0 || type >= num_species) {
                    throw std::runtime_error("site_type fuori intervallo in " + path);
                }
                if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])) {
                    throw std::runtime_error("coordinata non finita al frame " + std::to_string(f));
                }
                if (f == 0) {
                    set.site_types.push_back(type);
                    set.site_mol.push_back(m);
                } else if (site_cursor >= set.num_sites ||
                           set.site_types[site_cursor] != type ||
                           set.site_mol[site_cursor] != m) {
                    throw std::runtime_error("topologia diversa fra i frame di " + path);
                }
                set.positions.insert(set.positions.end(), {xyz[0], xyz[1], xyz[2]});
                ++site_cursor;
            }
        }
        if (site_cursor != set.num_sites) {
            throw std::runtime_error("num_total_sites incoerente al frame " + std::to_string(f));
        }
    }
    char trailing = 0;
    if (file.read(&trailing, 1)) throw std::runtime_error("byte in eccesso in coda a " + path);
    std::cout << "[INFO] " << path << ": " << set.num_frames << " frame, "
              << set.num_sites << " siti\n";
    return set;
}

// Tensori sul device, pronti per il batching.
struct DeviceSet {
    torch::Tensor pos;        // [F, N, 3]
    torch::Tensor box;        // [F, 3]
    torch::Tensor types;      // [N]
    torch::Tensor inter_mol;  // [N, N] bool: coppia di corpi diversi
    int64_t frames = 0;
    int64_t sites = 0;
};

static DeviceSet to_device(const PositionSet& set, torch::Device device, torch::Dtype dtype) {
    DeviceSet d;
    d.frames = set.num_frames;
    d.sites = set.num_sites;
    d.pos = torch::from_blob(const_cast<float*>(set.positions.data()),
                             {set.num_frames, set.num_sites, 3}, torch::kFloat32)
                .clone().to(dtype).to(device);
    d.box = torch::from_blob(const_cast<float*>(set.boxes.data()),
                             {set.num_frames, 3}, torch::kFloat32)
                .clone().to(dtype).to(device);
    d.types = torch::tensor(set.site_types, torch::kInt64).to(device);
    auto mol = torch::tensor(set.site_mol, torch::kInt64).to(device);
    d.inter_mol = mol.unsqueeze(1) != mol.unsqueeze(0);
    return d;
}

// ---------------------------------------------------------------------
// Energia della rete su un gruppo di frame.  Stessi archi di train_painn e
// del plugin: coppie di siti di corpi DIVERSI entro il cutoff, immagine
// minima, r_ij = r_i - r_j.
// ---------------------------------------------------------------------
static torch::Tensor batch_energy(PaiNNModel& model, const DeviceSet& set,
                                  const torch::Tensor& frame_idx, double cutoff) {
    const int64_t B = frame_idx.size(0);
    const int64_t N = set.sites;
    auto pos = set.pos.index_select(0, frame_idx);                // [B,N,3]
    auto box = set.box.index_select(0, frame_idx).view({B, 1, 1, 3});
    torch::Tensor edge_index, r_ij;
    {
        torch::NoGradGuard no_grad;
        auto diff = pos.unsqueeze(2) - pos.unsqueeze(1);          // [B,N,N,3]
        diff = diff - box * torch::round(diff / box);
        auto d2 = (diff * diff).sum(-1);
        auto mask = (d2 <= cutoff * cutoff) & set.inter_mol.unsqueeze(0);
        auto nz = mask.nonzero();                                 // [E,3] (b,i,j)
        auto b = nz.select(1, 0), i = nz.select(1, 1), j = nz.select(1, 2);
        edge_index = torch::stack({b * N + i, b * N + j});
        r_ij = diff.index({b, i, j});
    }
    auto types = set.types.repeat({B});
    auto batch_indices = torch::arange(B, types.options()).repeat_interleave(N);
    return model->forward_with_rij(types, r_ij, edge_index, batch_indices);  // [B]
}

// Energia su tutti i frame indicati, senza grafo, a blocchi.
static torch::Tensor energies_no_grad(PaiNNModel& model, const DeviceSet& set,
                                      const torch::Tensor& frame_idx, double cutoff,
                                      int64_t chunk) {
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> parts;
    const int64_t total = frame_idx.size(0);
    for (int64_t s = 0; s < total; s += chunk) {
        const int64_t e = std::min(total, s + chunk);
        parts.push_back(batch_energy(model, set, frame_idx.slice(0, s, e), cutoff)
                            .to(torch::kFloat64).cpu());
    }
    return torch::cat(parts);
}

static double log_mean_exp(const torch::Tensor& x) {
    return (torch::logsumexp(x, 0) - std::log(static_cast<double>(x.size(0)))).item<double>();
}

static std::vector<torch::Tensor> snapshot_parameters(PaiNNModel& model) {
    std::vector<torch::Tensor> copy;
    for (const auto& p : model->parameters()) copy.push_back(p.detach().clone());
    return copy;
}

static void restore_parameters(PaiNNModel& model, const std::vector<torch::Tensor>& saved) {
    torch::NoGradGuard no_grad;
    auto params = model->parameters();
    if (params.size() != saved.size()) throw std::runtime_error("snapshot incoerente");
    for (std::size_t k = 0; k < params.size(); ++k) params[k].copy_(saved[k]);
}

static void zero_readout_output(PaiNNModel& model) {
    torch::NoGradGuard no_grad;
    auto last = model->readout->ptr<torch::nn::LinearImpl>(2);
    last->weight.zero_();
    last->bias.zero_();
}

static void usage() {
    std::cerr <<
        "uso: train_relent --config C.json --aa AA.bin --out M.pt (--in M0.pt | --zero-init)\n"
        "       [--cg CG.bin] [--steps 100] [--lr 1e-3] [--weight-decay 0] [--grad-clip 0]\n"
        "       [--kT 2.49] [--batch-aa 8] [--batch-cg 8] [--eval-batch 8]\n"
        "       [--ess-min 0.5] [--holdout-frac 0.2] [--monitor-frames 256]\n"
        "       [--eval-every 10] [--energy-scale kT] [--seed 42] [--device auto]\n"
        "       [--report R.json] [--check-gradient]\n";
}

static int run(int argc, char* argv[]) {
    std::string config_path, aa_path, cg_path, in_path, out_path, report_path;
    std::string device_name = "auto";
    bool zero_init = false, check_gradient = false;
    int steps = 100, batch_aa = 8, batch_cg = 8, eval_batch = 8, eval_every = 10;
    int monitor_frames = 256, seed = 42;
    double lr = 1e-3, weight_decay = 0.0, grad_clip = 0.0, kT = 2.49;
    double ess_min = 0.5, holdout_frac = 0.2, energy_scale = -1.0;

    for (int k = 1; k < argc; ++k) {
        const std::string a = argv[k];
        auto next = [&]() -> std::string {
            if (k + 1 >= argc) { usage(); throw std::runtime_error("manca il valore di " + a); }
            return argv[++k];
        };
        if (a == "--config") config_path = next();
        else if (a == "--aa") aa_path = next();
        else if (a == "--cg") cg_path = next();
        else if (a == "--in") in_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--report") report_path = next();
        else if (a == "--device") device_name = next();
        else if (a == "--zero-init") zero_init = true;
        else if (a == "--check-gradient") check_gradient = true;
        else if (a == "--steps") steps = std::stoi(next());
        else if (a == "--batch-aa") batch_aa = std::stoi(next());
        else if (a == "--batch-cg") batch_cg = std::stoi(next());
        else if (a == "--eval-batch") eval_batch = std::stoi(next());
        else if (a == "--eval-every") eval_every = std::stoi(next());
        else if (a == "--monitor-frames") monitor_frames = std::stoi(next());
        else if (a == "--seed") seed = std::stoi(next());
        else if (a == "--lr") lr = std::stod(next());
        else if (a == "--weight-decay") weight_decay = std::stod(next());
        else if (a == "--grad-clip") grad_clip = std::stod(next());
        else if (a == "--kT") kT = std::stod(next());
        else if (a == "--ess-min") ess_min = std::stod(next());
        else if (a == "--holdout-frac") holdout_frac = std::stod(next());
        else if (a == "--energy-scale") energy_scale = std::stod(next());
        else { usage(); std::cerr << "[ERROR] opzione sconosciuta: " << a << "\n"; return 2; }
    }
    if (config_path.empty() || aa_path.empty() || (out_path.empty() && !check_gradient)) {
        usage();
        return 2;
    }
    if (zero_init == !in_path.empty()) {
        std::cerr << "[ERROR] serve esattamente una fra --in e --zero-init\n";
        return 2;
    }
    if ((steps > 0 || check_gradient) && cg_path.empty()) {
        std::cerr << "[ERROR] --cg e' necessario per --steps > 0 e per --check-gradient\n";
        return 2;
    }
    if (!(kT > 0.0) || !(ess_min > 0.0 && ess_min < 1.0) ||
        !(holdout_frac >= 0.0 && holdout_frac < 1.0) || batch_aa <= 0 || batch_cg <= 0 ||
        eval_batch <= 0 || eval_every <= 0 || steps < 0) {
        std::cerr << "[ERROR] parametri fuori intervallo\n";
        return 2;
    }
    if (!out_path.empty() && std::filesystem::exists(out_path)) {
        std::cerr << "[ERROR] il modello di uscita esiste gia': " << out_path << "\n";
        return 2;
    }
    const double beta = 1.0 / kT;
    torch::manual_seed(seed);

    torch::Device device(torch::kCPU);
    if (check_gradient || device_name == "cpu") {
        device = torch::Device(torch::kCPU);
    } else if (device_name == "cuda" || (device_name == "auto" && torch::cuda::is_available())) {
        device = torch::Device(torch::kCUDA);
    } else if (device_name == "mps" || (device_name == "auto" && torch::mps::is_available())) {
        device = torch::Device(torch::kMPS);
    }
    std::cout << "[INFO] device: " << device << "\n";

    // --- architettura: la stessa config del force matching e del plugin ---
    std::ifstream cfg_file(config_path);
    if (!cfg_file.is_open()) { std::cerr << "[ERROR] config illeggibile\n"; return 2; }
    json cfg = json::parse(cfg_file);
    const std::string variant = cfg.value("architecture_variant", std::string());
    if (variant != PAINN_ARCHITECTURE_VARIANT) {
        std::cerr << "[ERROR] train_relent supporta solo la variante "
                  << PAINN_ARCHITECTURE_VARIANT << ", non '" << variant << "'\n";
        return 2;
    }
    if (cfg.value("ordered_geometry_nodes", 0) > 0) {
        std::cerr << "[ERROR] la testa a geometria ordinata non e' supportata\n";
        return 2;
    }
    const int num_species = cfg.at("num_species").get<int>();
    const int dim = cfg.at("hidden_channels").get<int>();
    const int layers = cfg.at("n_layers").get<int>();
    const int num_rbf = cfg.at("num_rbf").get<int>();
    const double cutoff = cfg.at("cutoff").get<double>();
    const double toxvaerd_alpha = cfg.value("toxvaerd_alpha", 0.1);

    PaiNNModel model(num_species, dim, layers, num_rbf, cutoff, toxvaerd_alpha);
    if (!in_path.empty()) {
        torch::load(model, in_path);
        std::cout << "[INFO] modello di partenza: " << in_path << "\n";
    } else {
        zero_readout_output(model);
        const double scale = energy_scale > 0.0 ? energy_scale : kT;
        torch::NoGradGuard no_grad;
        model->energy_scale.fill_(scale);
        std::cout << "[INFO] inizializzazione a zero: U_ML == 0, energy_scale = "
                  << scale << " kJ/mol\n";
    }
    const torch::Dtype dtype = check_gradient ? torch::kFloat64 : torch::kFloat32;
    model->to(dtype);
    model->to(device);

    const auto t_start = std::chrono::steady_clock::now();
    json report = {
        {"schema", "mlcg_relent_v1"},
        {"config", config_path}, {"aa", aa_path}, {"cg", cg_path},
        {"model_in", in_path}, {"model_out", out_path}, {"zero_init", zero_init},
        {"kT_kj_mol", kT}, {"lr", lr}, {"weight_decay", weight_decay},
        {"grad_clip", grad_clip}, {"ess_min", ess_min}, {"batch_aa", batch_aa},
        {"batch_cg", batch_cg}, {"holdout_frac", holdout_frac}, {"seed", seed},
        {"energy_scale", model->energy_scale.item<double>()},
    };
    auto write_report = [&]() {
        if (report_path.empty()) return;
        std::ofstream out(report_path);
        out << std::setw(2) << report << "\n";
    };
    auto save_model = [&]() {
        const std::string tmp = out_path + ".tmp";
        torch::save(model, tmp);  // parametri e buffer (energy_scale compreso)
        std::filesystem::rename(tmp, out_path);
        std::cout << "[INFO] modello scritto: " << out_path << "\n";
    };

    PositionSet aa_raw = read_positions(aa_path, num_species);
    if (steps == 0 && !check_gradient) {
        report["steps_done"] = 0;
        report["stop_reason"] = "steps=0";
        save_model();
        write_report();
        return 0;
    }
    PositionSet cg_raw = read_positions(cg_path, num_species);
    if (cg_raw.site_types != aa_raw.site_types || cg_raw.site_mol != aa_raw.site_mol) {
        std::cerr << "[ERROR] AA e CG non hanno la stessa topologia di siti\n";
        return 2;
    }
    DeviceSet aa = to_device(aa_raw, device, dtype);
    DeviceSet cg = to_device(cg_raw, device, dtype);
    aa_raw.positions.clear(); aa_raw.positions.shrink_to_fit();
    cg_raw.positions.clear(); cg_raw.positions.shrink_to_fit();

    // =================================================================
    // Verifica del gradiente: il surrogato beta(<U>_AA - sum_i w_i U_i), con
    // w detached, deve avere come gradiente quello di
    //   F(theta) = beta <U>_AA + ln <exp(-beta (U - U_0))>_CG
    // anche lontano da theta_0, dove i pesi non sono uniformi.
    // =================================================================
    if (check_gradient) {
        const int64_t na = std::min<int64_t>(aa.frames, 6);
        const int64_t nc = std::min<int64_t>(cg.frames, 6);
        auto ia = torch::arange(na, torch::kInt64);
        auto ic = torch::arange(nc, torch::kInt64);
        auto perturb = [&](double amplitude) {
            torch::NoGradGuard no_grad;
            for (auto& p : model->parameters()) p.add_(torch::randn_like(p) * amplitude);
        };
        // Un modello non banale (con --zero-init l'energia sarebbe nulla).
        perturb(0.3);
        {
            // Scala l'energia a una dispersione di ~2 kT fra i frame: pesi di
            // ripesatura informativi, ne' uniformi ne' degeneri.
            auto u = energies_no_grad(model, cg, ic, cutoff, nc);
            const double spread = u.std().item<double>();
            torch::NoGradGuard no_grad;
            if (spread > 0.0) model->energy_scale.mul_(2.0 * kT / spread);
        }
        // Coerenza del batching: stessa energia a blocchi di 1 e tutti insieme.
        auto u_one = energies_no_grad(model, cg, ic, cutoff, 1);
        auto u_all = energies_no_grad(model, cg, ic, cutoff, nc);
        const double batch_err = (u_one - u_all).abs().max().item<double>();
        const double u_spread = u_all.std().item<double>();
        std::cout << "[CHECK] batching: max|U(1) - U(" << nc << ")| = " << batch_err
                  << " (dispersione di U fra i frame: " << u_spread << " kJ/mol)\n";

        // theta_0 = questo modello; theta = theta_0 + perturbazione, cosi' i
        // pesi di ripesatura non sono uniformi.
        auto u0 = u_all;
        perturb(0.005);
        auto objective = [&]() {
            auto ua = energies_no_grad(model, aa, ia, cutoff, na);
            auto uc = energies_no_grad(model, cg, ic, cutoff, nc);
            return beta * ua.mean().item<double>() + log_mean_exp(-beta * (uc - u0));
        };
        model->zero_grad();
        auto uc_now = energies_no_grad(model, cg, ic, cutoff, nc);
        auto w = torch::softmax(-beta * (uc_now - u0), 0).to(dtype);
        auto ua_g = batch_energy(model, aa, ia, cutoff);
        auto uc_g = batch_energy(model, cg, ic, cutoff);
        auto surrogate = beta * (ua_g.mean() - (w * uc_g).sum());
        surrogate.backward();
        std::vector<torch::Tensor> dir;
        double analytic = 0.0, norm2 = 0.0;
        for (auto& p : model->parameters()) {
            auto d = torch::randn_like(p);
            norm2 += (d * d).sum().item<double>();
            dir.push_back(d);
        }
        const double inv_norm = 1.0 / std::sqrt(norm2);
        auto params = model->parameters();
        for (std::size_t k = 0; k < params.size(); ++k) {
            dir[k] = dir[k] * inv_norm;
            if (params[k].grad().defined()) {
                analytic += (params[k].grad() * dir[k]).sum().item<double>();
            }
        }
        auto shift = [&](double s) {
            torch::NoGradGuard no_grad;
            for (std::size_t k = 0; k < params.size(); ++k) params[k].add_(dir[k] * s);
        };
        // Differenze centrali a piu' passi, con estrapolazione di Richardson
        // fra eps ed eps/2 (l'errore di troncamento va come eps^2): su un
        // sistema grande la curvatura in theta e' grande e un passo solo non
        // separa il troncamento dall'arrotondamento.
        auto central = [&](double eps) {
            shift(+eps); const double fp = objective();
            shift(-2 * eps); const double fm = objective();
            shift(+eps);
            return (fp - fm) / (2 * eps);
        };
        const double ess = 1.0 / (w * w).sum().item<double>() / static_cast<double>(nc);
        std::cout << std::setprecision(10)
                  << "[CHECK] ESS/N = " << ess << " (pesi non uniformi se < 1)\n"
                  << "[CHECK] derivata direzionale analitica: " << analytic << "\n";
        double best_rel = std::numeric_limits<double>::infinity();
        for (double eps : {1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6}) {
            const double n1 = central(eps);
            const double n2 = central(eps / 2);
            const double richardson = (4.0 * n2 - n1) / 3.0;
            const double rel_raw = std::abs(analytic - n2) / std::max(1e-12, std::abs(n2));
            const double rel_rich =
                std::abs(analytic - richardson) / std::max(1e-12, std::abs(richardson));
            best_rel = std::min(best_rel, rel_rich);
            std::cout << "[CHECK]   eps " << std::setw(7) << eps
                      << "  centrale " << n2 << " (err. rel. " << rel_raw << ")"
                      << "  Richardson " << richardson << " (err. rel. " << rel_rich << ")\n";
        }
        std::cout << "[CHECK] miglior errore relativo (Richardson): " << best_rel << "\n";
        const bool ok = best_rel < 1e-6 && batch_err < 1e-9 * std::max(1.0, u_spread) && u_spread > 0.0;
        std::cout << (ok ? "[OK] gradiente dell'entropia relativa verificato\n"
                         : "[FAIL] gradiente o batching non coerenti\n");
        return ok ? 0 : 1;
    }

    // --- divisione AA: la coda della traiettoria resta fuori (holdout) ---
    const int64_t n_hold = static_cast<int64_t>(std::floor(holdout_frac * aa.frames));
    const int64_t n_train = aa.frames - n_hold;
    if (n_train < batch_aa) { std::cerr << "[ERROR] troppo pochi frame AA\n"; return 2; }
    auto aa_train_idx = torch::arange(n_train, torch::kInt64);
    auto aa_hold_idx = torch::arange(n_train, aa.frames, torch::kInt64);
    torch::Tensor aa_mon_idx = std::get<0>(
        torch::randperm(n_train, torch::kInt64)
            .slice(0, 0, std::min<int64_t>(n_train, monitor_frames))
            .sort());
    auto cg_idx = torch::arange(cg.frames, torch::kInt64);
    auto on_dev = [&](const torch::Tensor& t) { return t.to(device); };

    auto u0_cg = energies_no_grad(model, cg, on_dev(cg_idx), cutoff, eval_batch);
    auto u0_mon = energies_no_grad(model, aa, on_dev(aa_mon_idx), cutoff, eval_batch);
    auto u0_hold = n_hold > 0
        ? energies_no_grad(model, aa, on_dev(aa_hold_idx), cutoff, eval_batch)
        : torch::zeros({0}, torch::kFloat64);
    std::cout << "[INFO] frame AA: " << n_train << " allenamento (" << aa_mon_idx.size(0)
              << " di controllo), " << n_hold << " holdout; frame CG: " << cg.frames << "\n";

    torch::optim::AdamW optimizer(
        model->parameters(), torch::optim::AdamWOptions(lr).weight_decay(weight_decay));

    struct Eval { double ess, ds_train, ds_hold, gap; };
    auto evaluate = [&](const torch::Tensor& u_cg) {
        Eval e{};
        auto du = u_cg - u0_cg;
        auto w = torch::softmax(-beta * du, 0);
        e.ess = 1.0 / (w * w).sum().item<double>() / static_cast<double>(cg.frames);
        const double z_term = log_mean_exp(-beta * du);
        auto u_mon = energies_no_grad(model, aa, on_dev(aa_mon_idx), cutoff, eval_batch);
        e.ds_train = beta * (u_mon - u0_mon).mean().item<double>() + z_term;
        e.ds_hold = std::numeric_limits<double>::quiet_NaN();
        if (n_hold > 0) {
            auto u_hold = energies_no_grad(model, aa, on_dev(aa_hold_idx), cutoff, eval_batch);
            e.ds_hold = beta * (u_hold - u0_hold).mean().item<double>() + z_term;
        }
        e.gap = beta * (u_mon.mean().item<double>() - (w * u_cg).sum().item<double>());
        return e;
    };

    json history = json::array();
    double best_score = 0.0;  // Delta S di theta_0 e' zero per costruzione
    int best_step = 0;
    auto best_params = snapshot_parameters(model);
    auto prev_params = best_params;
    std::string stop_reason = "steps";
    int steps_done = 0;
    double last_ess = 1.0;

    std::cout << "[INFO] passo      ESS/N      dS_train      dS_hold     beta*gap\n";
    for (int step = 1; step <= steps; ++step) {
        auto u_cg = energies_no_grad(model, cg, on_dev(cg_idx), cutoff, eval_batch);
        auto w_cg = torch::softmax(-beta * (u_cg - u0_cg), 0);
        const double ess = 1.0 / (w_cg * w_cg).sum().item<double>() / static_cast<double>(cg.frames);
        last_ess = ess;
        if (ess < ess_min) {
            // L'ultimo passo ha portato theta troppo lontano dai campioni:
            // lo si annulla e si torna a simulare.
            restore_parameters(model, prev_params);
            steps_done = step - 2;  // lo stato ripristinato e' quello prima del passo step-1
            stop_reason = "ess";
            std::cout << "[INFO] ESS/N = " << ess << " < " << ess_min
                      << " al passo " << step << ": serve una nuova simulazione\n";
            break;
        }
        if ((step - 1) % eval_every == 0) {
            const Eval e = evaluate(u_cg);
            const double score = n_hold > 0 ? e.ds_hold : e.ds_train;
            std::cout << "[RELENT] " << std::setw(5) << (step - 1) << std::fixed
                      << std::setprecision(4) << std::setw(10) << e.ess
                      << std::setw(14) << e.ds_train << std::setw(13) << e.ds_hold
                      << std::setw(13) << e.gap << std::defaultfloat << "\n";
            history.push_back({{"step", step - 1}, {"ess", e.ess}, {"dS_train", e.ds_train},
                           {"dS_hold", e.ds_hold}, {"beta_gap", e.gap}});
            if (score < best_score) {
                best_score = score;
                best_step = step - 1;
                best_params = snapshot_parameters(model);
            }
        }
        prev_params = snapshot_parameters(model);

        optimizer.zero_grad();
        auto ia = on_dev(aa_train_idx.index_select(
            0, torch::randint(n_train, {batch_aa}, torch::kInt64)));
        // Campionamento per importanza dai pesi: media semplice sul batch.
        auto ic = on_dev(torch::multinomial(w_cg.to(torch::kFloat64), batch_cg, true));
        auto loss = beta * (batch_energy(model, aa, ia, cutoff).mean() -
                            batch_energy(model, cg, ic, cutoff).mean());
        loss.backward();
        if (grad_clip > 0.0) torch::nn::utils::clip_grad_norm_(model->parameters(), grad_clip);
        optimizer.step();
        steps_done = step;
    }
    // Stato finale (dopo l'ultimo passo, o quello prima del crollo dell'ESS):
    // lo si valuta sempre, perche' con eval_every > 1 potrebbe essere il migliore.
    {
        auto u_cg = energies_no_grad(model, cg, on_dev(cg_idx), cutoff, eval_batch);
        const Eval e = evaluate(u_cg);
        last_ess = e.ess;
        if (e.ess >= ess_min) {
            const double score = n_hold > 0 ? e.ds_hold : e.ds_train;
            history.push_back({{"step", steps_done}, {"ess", e.ess}, {"dS_train", e.ds_train},
                               {"dS_hold", e.ds_hold}, {"beta_gap", e.gap}, {"final", true}});
            std::cout << "[RELENT] " << std::setw(5) << steps_done << std::fixed
                      << std::setprecision(4) << std::setw(10) << e.ess
                      << std::setw(14) << e.ds_train << std::setw(13) << e.ds_hold
                      << std::setw(13) << e.gap << std::defaultfloat << "  (finale)\n";
            if (score < best_score) {
                best_score = score;
                best_step = steps_done;
                best_params = snapshot_parameters(model);
            }
        }
    }
    restore_parameters(model, best_params);
    std::cout << "[INFO] miglior passo " << best_step << " (dS "
              << (n_hold > 0 ? "holdout" : "train") << " = " << best_score
              << "), arresto: " << stop_reason << "\n";
    if (best_step == 0) {
        std::cout << "[WARNING] nessun passo abbassa S_rel: il modello resta theta_0\n";
    }

    save_model();
    report["steps_done"] = steps_done;
    report["stop_reason"] = stop_reason;
    report["ess_last"] = last_ess;
    report["best_step"] = best_step;
    report["best_dS"] = best_score;
    report["best_dS_source"] = n_hold > 0 ? "holdout" : "train";
    report["frames"] = {{"aa_train", n_train}, {"aa_holdout", n_hold},
                        {"aa_monitor", aa_mon_idx.size(0)}, {"cg", cg.frames}};
    report["log"] = history;
    report["seconds"] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    write_report();
    return 0;
}

int main(int argc, char* argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
