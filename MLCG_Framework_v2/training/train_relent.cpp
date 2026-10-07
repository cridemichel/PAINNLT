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
    torch::Tensor group;      // [N] gruppo (copia) di ogni sito, 0..G-1
    int64_t groups = 1;
    int64_t frames = 0;
    int64_t sites = 0;
};

// residues_per_copy > 0: i corpi (residui) consecutivi formano copie di
// residues_per_copy corpi ciascuna, e ogni copia e' un gruppo di ripesatura.
static DeviceSet to_device(const PositionSet& set, torch::Device device, torch::Dtype dtype,
                           int residues_per_copy) {
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
    const int64_t num_mol = set.site_mol.empty() ? 0 : set.site_mol.back() + 1;
    if (residues_per_copy > 0) {
        if (num_mol % residues_per_copy != 0) {
            throw std::runtime_error("il numero di corpi (" + std::to_string(num_mol) +
                                     ") non e' multiplo di --residues-per-copy");
        }
        d.groups = num_mol / residues_per_copy;
        d.group = torch::div(mol, residues_per_copy, "floor");
    } else {
        d.groups = 1;
        d.group = torch::zeros_like(mol);
    }
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
    // Energia per (frame, gruppo): la rete somma le energie di sito per
    // batch_indices, quindi basta indicizzare frame*G + gruppo.  I messaggi
    // restano quelli del sistema intero (archi fra copie compresi).
    const int64_t G = set.groups;
    auto batch_indices = (torch::arange(B, types.options()).repeat_interleave(N) * G +
                          set.group.repeat({B}));
    return model->forward_with_rij(types, r_ij, edge_index, batch_indices).view({B, G});
}

// Energia [frame, gruppo] su tutti i frame indicati, senza grafo, a blocchi.
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

// Somma sui gruppi di ln <exp(x)>_frame.  Con un gruppo solo e' il vecchio
// ln <exp(x)>; con piu' gruppi e' la stima fattorizzata (copie indipendenti).
static double log_mean_exp(const torch::Tensor& x) {
    auto xx = x.dim() == 1 ? x.unsqueeze(1) : x;
    return (torch::logsumexp(xx, 0) - std::log(static_cast<double>(xx.size(0)))).sum().item<double>();
}

// Pesi di ripesatura per gruppo (normalizzati sui frame, colonna per colonna)
// e ESS/N del gruppo peggiore.
static torch::Tensor group_weights(const torch::Tensor& log_w) {
    return torch::softmax(log_w, 0);
}
static double min_ess_fraction(const torch::Tensor& w) {
    auto ess = 1.0 / (w * w).sum(0);                   // [G]
    return (ess.min() / static_cast<double>(w.size(0))).item<double>();
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

static constexpr double R_KJ_MOL_K = 0.008314462618;  // kJ/(mol K)

static void usage() {
    std::cerr <<
        "uso: train_relent --config C.json --aa AA.bin --out M.pt (--in M0.pt | --zero-init)\n"
        "       [--cg CG.bin] [--steps 100] [--lr 1e-3] [--weight-decay 0] [--grad-clip 0]\n"
        "       [--kT 2.49] [--batch-aa 8] [--batch-cg 8] [--eval-batch 8]\n"
        "       [--ess-min 0.5] [--holdout-frac 0.2] [--monitor-frames 256]\n"
        "       [--eval-every 10] [--energy-scale kT] [--seed 42] [--device auto]\n"
        "       [--max-backtracks 6] [--residues-per-copy 0]\n"
        "       [--report R.json] [--check-gradient]\n"
        "       piu' temperature (config con \"thermo_heads\": true):\n"
        "       --system AA.bin,CG.bin,T_K[,peso] (ripetibile, al posto di --aa/--cg)\n"
        "       [--trainable all|thermo]\n";
}

static int run(int argc, char* argv[]) {
    // Output riga per riga anche quando stdout e' un file (log di Slurm):
    // altrimenti l'avanzamento compare solo a fine processo.
    std::cout << std::unitbuf;
    std::string config_path, aa_path, cg_path, in_path, out_path, report_path;
    std::string device_name = "auto";
    bool zero_init = false, check_gradient = false;
    int steps = 100, batch_aa = 8, batch_cg = 8, eval_batch = 8, eval_every = 10;
    int monitor_frames = 256, seed = 42, max_backtracks = 6, residues_per_copy = 0;
    double lr = 1e-3, weight_decay = 0.0, grad_clip = 0.0, kT = 2.49;
    double ess_min = 0.5, holdout_frac = 0.2, energy_scale = -1.0;
    std::vector<std::string> system_specs;   // --system AA.bin,CG.bin,T_K[,peso]
    std::string trainable = "all";            // all | thermo (solo readout_dT)

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
        else if (a == "--max-backtracks") max_backtracks = std::stoi(next());
        else if (a == "--residues-per-copy") residues_per_copy = std::stoi(next());
        else if (a == "--lr") lr = std::stod(next());
        else if (a == "--weight-decay") weight_decay = std::stod(next());
        else if (a == "--grad-clip") grad_clip = std::stod(next());
        else if (a == "--kT") kT = std::stod(next());
        else if (a == "--ess-min") ess_min = std::stod(next());
        else if (a == "--holdout-frac") holdout_frac = std::stod(next());
        else if (a == "--energy-scale") energy_scale = std::stod(next());
        else if (a == "--system") system_specs.push_back(next());
        else if (a == "--trainable") trainable = next();
        else { usage(); std::cerr << "[ERROR] opzione sconosciuta: " << a << "\n"; return 2; }
    }
    if (config_path.empty() || (aa_path.empty() && system_specs.empty()) ||
        (out_path.empty() && !check_gradient)) {
        usage();
        return 2;
    }
    if (zero_init == !in_path.empty()) {
        std::cerr << "[ERROR] serve esattamente una fra --in e --zero-init\n";
        return 2;
    }
    if (!system_specs.empty() && (!aa_path.empty() || !cg_path.empty())) {
        std::cerr << "[ERROR] --system esclude --aa/--cg\n";
        return 2;
    }
    if ((steps > 0 || check_gradient) && system_specs.empty() && cg_path.empty()) {
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

    const bool thermo = cfg.value("thermo_heads", false);
    const double thermo_T0 = cfg.value("thermo_T0", 300.0);
    PaiNNModel model(num_species, dim, layers, num_rbf, cutoff, toxvaerd_alpha,
                     0, 5, 160, 0.0, false, 1, false, thermo, thermo_T0);
    if (thermo) {
        std::cout << "[INFO] teste termodinamiche: U(T) = U(T0) + (T - T0) dU/dT, T0 = "
                  << thermo_T0 << " K\n";
    }
    if (!in_path.empty()) {
        const bool promoted = load_painn_weights(model, in_path, dim);
        std::cout << "[INFO] modello di partenza: " << in_path
                  << (promoted ? " (senza teste termodinamiche: readout_dT inizializzata a zero)" : "")
                  << "\n";
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
        {"thermo_heads", thermo}, {"thermo_T0_K", thermo_T0},
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

    // =================================================================
    // Sistemi (AA, CG, T, peso).  Con --system ripetuto si allena su piu'
    // temperature insieme -- serve un modello con teste termodinamiche, perche'
    // con dati a una sola T solo U = H - T S e' determinata, non H e S
    // separatamente.  Senza --system: un solo sistema da --aa/--cg/--kT,
    // esattamente come prima.  L'obiettivo e' sum_k w_k S_rel,k: ogni sistema
    // ha la sua ripesatura, il suo holdout e la sua soglia di ESS.
    // =================================================================
    struct System {
        std::string aa_path, cg_path;
        double T = 0.0, beta = 0.0, weight = 1.0;
        DeviceSet aa, cg;
        int64_t n_train = 0, n_hold = 0;
        torch::Tensor aa_train_idx, aa_hold_idx, aa_mon_idx, cg_idx;
        torch::Tensor u0_cg, u0_mon, u0_hold;
    };
    std::vector<System> systems;
    if (system_specs.empty()) {
        System s;
        s.aa_path = aa_path; s.cg_path = cg_path;
        s.T = kT / R_KJ_MOL_K; s.beta = beta;
        systems.push_back(s);
    } else {
        for (const auto& spec : system_specs) {
            std::vector<std::string> f;
            std::size_t start = 0;
            while (true) {
                auto pos = spec.find(',', start);
                f.push_back(spec.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
            if (f.size() < 3 || f.size() > 4) {
                throw std::runtime_error("--system vuole AA.bin,CG.bin,T_K[,peso]: " + spec);
            }
            System s;
            s.aa_path = f[0]; s.cg_path = f[1];
            s.T = std::stod(f[2]);
            s.weight = f.size() == 4 ? std::stod(f[3]) : 1.0;
            if (!(s.T > 0.0) || !(s.weight > 0.0)) throw std::runtime_error("T e peso devono essere positivi: " + spec);
            s.beta = 1.0 / (R_KJ_MOL_K * s.T);
            systems.push_back(s);
        }
    }
    if (!model->has_thermo_heads()) {
        for (const auto& s : systems) {
            if (std::abs(s.T - systems[0].T) > 1e-9 * systems[0].T) {
                std::cerr << "[ERROR] sistemi a temperature diverse richiedono \"thermo_heads\": true nella config\n";
                return 2;
            }
        }
    }
    report["systems"] = json::array();
    for (const auto& s : systems) {
        report["systems"].push_back({{"aa", s.aa_path}, {"cg", s.cg_path}, {"T_K", s.T},
                                     {"beta_mol_kJ", s.beta}, {"weight", s.weight}});
    }
    if (steps == 0 && !check_gradient) {
        report["steps_done"] = 0;
        report["stop_reason"] = "steps=0";
        save_model();
        write_report();
        return 0;
    }
    for (auto& s : systems) {
        if (s.cg_path.empty()) { std::cerr << "[ERROR] manca il CG del sistema " << s.aa_path << "\n"; return 2; }
        PositionSet aa_raw = read_positions(s.aa_path, num_species);
        PositionSet cg_raw = read_positions(s.cg_path, num_species);
        if (cg_raw.site_types != aa_raw.site_types || cg_raw.site_mol != aa_raw.site_mol) {
            std::cerr << "[ERROR] AA e CG non hanno la stessa topologia di siti: " << s.aa_path << "\n";
            return 2;
        }
        s.aa = to_device(aa_raw, device, dtype, residues_per_copy);
        s.cg = to_device(cg_raw, device, dtype, residues_per_copy);
        std::cout << "[INFO] sistema T = " << s.T << " K (peso " << s.weight << "): "
                  << s.aa.frames << " frame AA, " << s.cg.frames << " frame CG, "
                  << s.cg.groups << " gruppi di ripesatura\n";
    }
    report["groups"] = systems[0].cg.groups;

    auto set_T = [&](const System& s) {
        if (model->has_thermo_heads()) model->set_temperature(s.T);
    };
    auto energy_g = [&](const System& s, const DeviceSet& set, const torch::Tensor& idx) {
        set_T(s);
        return batch_energy(model, set, idx, cutoff);
    };
    auto energy_ng = [&](const System& s, const DeviceSet& set, const torch::Tensor& idx, int64_t chunk) {
        set_T(s);
        return energies_no_grad(model, set, idx, cutoff, chunk);
    };

    // =================================================================
    // Verifica del gradiente: il surrogato sum_k w_k beta_k(<U_k>_AA -
    // sum_i w_ik U_ik), con w detached, deve avere come gradiente quello di
    //   F(theta) = sum_k w_k [beta_k <U_k>_AA + ln <exp(-beta_k (U_k - U_k0))>_CG]
    // anche lontano da theta_0, dove i pesi non sono uniformi.
    // =================================================================
    if (check_gradient) {
        std::vector<torch::Tensor> ia_k, ic_k;
        for (const auto& s : systems) {
            ia_k.push_back(torch::arange(std::min<int64_t>(s.aa.frames, 6), torch::kInt64));
            ic_k.push_back(torch::arange(std::min<int64_t>(s.cg.frames, 6), torch::kInt64));
        }
        auto perturb = [&](double amplitude) {
            torch::NoGradGuard no_grad;
            for (auto& p : model->parameters()) {
                double scale = p.pow(2).mean().sqrt().item<double>();
                if (!(scale > 0.0)) scale = 0.1;
                p.add_(torch::randn_like(p) * (amplitude * scale));
            }
        };
        perturb(0.3);
        {
            const System& s = systems[0];
            const int64_t nc = ic_k[0].size(0);
            auto u = energy_ng(s, s.cg, ic_k[0], nc).sum(1);
            const double spread = u.std().item<double>();
            torch::NoGradGuard no_grad;
            if (spread > 0.0) model->energy_scale.mul_(2.0 / (s.beta * spread));
        }
        double batch_err = 0.0, u_spread = 0.0;
        std::vector<torch::Tensor> u0_k;
        for (std::size_t k = 0; k < systems.size(); ++k) {
            const System& s = systems[k];
            const int64_t nc = ic_k[k].size(0);
            auto u_one = energy_ng(s, s.cg, ic_k[k], 1);
            auto u_all = energy_ng(s, s.cg, ic_k[k], nc);
            batch_err = std::max(batch_err, (u_one - u_all).abs().max().item<double>());
            u_spread = std::max(u_spread, u_all.sum(1).std().item<double>());
        }
        std::cout << "[CHECK] batching: max|U(1) - U(tutti)| = " << batch_err
                  << " (dispersione di U fra i frame: " << u_spread << " kJ/mol)\n";
        for (std::size_t k = 0; k < systems.size(); ++k) {
            u0_k.push_back(energy_ng(systems[k], systems[k].cg, ic_k[k], ic_k[k].size(0)));
        }
        perturb(0.15);
        auto objective = [&]() {
            double total = 0.0;
            for (std::size_t k = 0; k < systems.size(); ++k) {
                const System& s = systems[k];
                auto ua = energy_ng(s, s.aa, ia_k[k], ia_k[k].size(0));
                auto uc = energy_ng(s, s.cg, ic_k[k], ic_k[k].size(0));
                total += s.weight * (s.beta * ua.sum(1).mean().item<double>() +
                                     log_mean_exp(-s.beta * (uc - u0_k[k])));
            }
            return total;
        };
        model->zero_grad();
        double ess = 1.0;
        for (std::size_t k = 0; k < systems.size(); ++k) {
            const System& s = systems[k];
            auto uc_now = energy_ng(s, s.cg, ic_k[k], ic_k[k].size(0));
            auto w = torch::softmax(-s.beta * (uc_now - u0_k[k]), 0).to(dtype);
            ess = std::min(ess, min_ess_fraction(w));
            auto ua_g = energy_g(s, s.aa, ia_k[k]);
            auto uc_g = energy_g(s, s.cg, ic_k[k]);
            auto surrogate = s.weight * s.beta * (ua_g.sum(1).mean() - (w * uc_g).sum());
            surrogate.backward();
        }
        auto params = model->parameters();
        auto mark = [&](torch::nn::Sequential seq) {
            std::vector<bool> in(params.size(), false);
            if (!seq) return in;
            std::vector<const void*> ptrs;
            for (const auto& p : seq->parameters()) ptrs.push_back(p.unsafeGetTensorImpl());
            for (std::size_t k = 0; k < params.size(); ++k) {
                in[k] = std::find(ptrs.begin(), ptrs.end(), params[k].unsafeGetTensorImpl()) != ptrs.end();
            }
            return in;
        };
        const auto in_readout = mark(model->readout);
        const auto in_readout_dT = mark(model->readout_dT);
        auto directional_check = [&](const std::string& label, const std::vector<bool>* subset) {
            std::vector<torch::Tensor> dir;
            double norm2 = 0.0;
            for (std::size_t k = 0; k < params.size(); ++k) {
                auto d = torch::randn_like(params[k]);
                if (subset != nullptr && !(*subset)[k]) d.zero_();
                norm2 += (d * d).sum().item<double>();
                dir.push_back(d);
            }
            const double inv_norm = 1.0 / std::sqrt(norm2);
            double analytic = 0.0;
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
            auto central = [&](double eps) {
                shift(+eps); const double fp = objective();
                shift(-2 * eps); const double fm = objective();
                shift(+eps);
                return (fp - fm) / (2 * eps);
            };
            std::cout << "[CHECK] direzione " << label << ": derivata analitica " << analytic << "\n";
            double best_rich = std::numeric_limits<double>::infinity();
            double best_any = std::numeric_limits<double>::infinity();
            for (double eps : {1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6}) {
                const double n1 = central(eps);
                const double n2 = central(eps / 2);
                const double richardson = (4.0 * n2 - n1) / 3.0;
                const double rel_raw = std::abs(analytic - n2) / std::max(1e-12, std::abs(n2));
                const double rel_rich =
                    std::abs(analytic - richardson) / std::max(1e-12, std::abs(richardson));
                best_rich = std::min(best_rich, rel_rich);
                best_any = std::min({best_any, rel_rich, rel_raw});
                std::cout << "[CHECK]   eps " << std::setw(7) << eps
                          << "  centrale " << n2 << " (err. rel. " << rel_raw << ")"
                          << "  Richardson " << richardson << " (err. rel. " << rel_rich << ")\n";
            }
            std::cout << "[CHECK]   miglior errore relativo: Richardson " << best_rich
                      << ", qualunque passo " << best_any << "\n";
            return std::make_pair(best_rich, best_any);
        };
        std::cout << std::setprecision(10)
                  << "[CHECK] ESS/N = " << ess << " (pesi non uniformi se < 1)\n";
        const auto readout = directional_check("solo readout", &in_readout);
        bool thermo_ok = true;
        if (model->has_thermo_heads()) {
            const auto th = directional_check("solo testa termodinamica (readout_dT)", &in_readout_dT);
            thermo_ok = th.first < 1e-6;
        }
        const auto all = directional_check("tutti i parametri", nullptr);
        const bool batching_ok = batch_err < 1e-9 * std::max(1.0, u_spread) && u_spread > 0.0;
        const bool ok = batching_ok && readout.first < 1e-6 && all.first < 1e-6 && thermo_ok;
        std::cout << (ok ? "[OK] gradiente dell'entropia relativa verificato\n"
                         : "[FAIL] gradiente o batching non coerenti\n");
        return ok ? 0 : 1;
    }

    // --- per sistema: divisione AA (la coda resta fuori) ed energie di theta_0 ---
    auto on_dev = [&](const torch::Tensor& t) { return t.to(device); };
    for (auto& s : systems) {
        s.n_hold = static_cast<int64_t>(std::floor(holdout_frac * s.aa.frames));
        s.n_train = s.aa.frames - s.n_hold;
        if (s.n_train < batch_aa) { std::cerr << "[ERROR] troppo pochi frame AA in " << s.aa_path << "\n"; return 2; }
        s.aa_train_idx = torch::arange(s.n_train, torch::kInt64);
        s.aa_hold_idx = torch::arange(s.n_train, s.aa.frames, torch::kInt64);
        s.aa_mon_idx = std::get<0>(
            torch::randperm(s.n_train, torch::kInt64)
                .slice(0, 0, std::min<int64_t>(s.n_train, monitor_frames))
                .sort());
        s.cg_idx = torch::arange(s.cg.frames, torch::kInt64);
        s.u0_cg = energy_ng(s, s.cg, on_dev(s.cg_idx), eval_batch);
        s.u0_mon = energy_ng(s, s.aa, on_dev(s.aa_mon_idx), eval_batch).sum(1);
        s.u0_hold = s.n_hold > 0
            ? energy_ng(s, s.aa, on_dev(s.aa_hold_idx), eval_batch).sum(1)
            : torch::zeros({0}, torch::kFloat64);
        std::cout << "[INFO] T = " << s.T << " K: frame AA " << s.n_train << " allenamento ("
                  << s.aa_mon_idx.size(0) << " di controllo), " << s.n_hold
                  << " holdout; frame CG: " << s.cg.frames << "\n";
    }

    // Parametri allenati: tutti, oppure la sola testa termodinamica (il resto
    // del modello, per esempio quello allenato a 300 K, resta fermo).
    std::vector<torch::Tensor> train_params;
    if (trainable == "all") {
        train_params = model->parameters();
    } else if (trainable == "thermo") {
        if (!model->has_thermo_heads()) { std::cerr << "[ERROR] --trainable thermo senza teste termodinamiche\n"; return 2; }
        train_params = model->readout_dT->parameters();
        for (auto& p : model->parameters()) p.set_requires_grad(false);
        for (auto& p : train_params) p.set_requires_grad(true);
    } else {
        std::cerr << "[ERROR] --trainable vuole all oppure thermo\n";
        return 2;
    }
    report["trainable"] = trainable;
    torch::optim::AdamW optimizer(
        train_params, torch::optim::AdamWOptions(lr).weight_decay(weight_decay));

    struct Eval { double ess, ds_train, ds_hold, gap; };
    // Energie CG correnti per sistema (senza grafo).
    auto current_cg = [&]() {
        std::vector<torch::Tensor> u;
        for (auto& s : systems) u.push_back(energy_ng(s, s.cg, on_dev(s.cg_idx), eval_batch));
        return u;
    };
    auto evaluate_one = [&](System& s, const torch::Tensor& u_cg) {
        Eval e{};
        auto du = u_cg - s.u0_cg;
        auto w = torch::softmax(-s.beta * du, 0);
        e.ess = min_ess_fraction(w);
        const double z_term = log_mean_exp(-s.beta * du);
        auto u_mon = energy_ng(s, s.aa, on_dev(s.aa_mon_idx), eval_batch).sum(1);
        e.ds_train = s.beta * (u_mon - s.u0_mon).mean().item<double>() + z_term;
        e.ds_hold = std::numeric_limits<double>::quiet_NaN();
        if (s.n_hold > 0) {
            auto u_hold = energy_ng(s, s.aa, on_dev(s.aa_hold_idx), eval_batch).sum(1);
            e.ds_hold = s.beta * (u_hold - s.u0_hold).mean().item<double>() + z_term;
        }
        e.gap = s.beta * (u_mon.mean().item<double>() - (w * u_cg).sum().item<double>());
        return e;
    };
    auto evaluate = [&](const std::vector<torch::Tensor>& u_cg, json* per_system) {
        Eval tot{1.0, 0.0, 0.0, 0.0};
        for (std::size_t k = 0; k < systems.size(); ++k) {
            const Eval e = evaluate_one(systems[k], u_cg[k]);
            const double w = systems[k].weight;
            tot.ess = std::min(tot.ess, e.ess);
            tot.ds_train += w * e.ds_train;
            tot.ds_hold += w * e.ds_hold;
            tot.gap += w * e.gap;
            if (per_system) {
                per_system->push_back({{"T_K", systems[k].T}, {"ess", e.ess}, {"dS_train", e.ds_train},
                                       {"dS_hold", e.ds_hold}, {"beta_gap", e.gap}});
            }
        }
        return tot;
    };
    const bool has_hold = std::all_of(systems.begin(), systems.end(),
                                      [](const System& s) { return s.n_hold > 0; });

    json history = json::array();
    double best_score = 0.0;  // Delta S di theta_0 e' zero per costruzione
    int best_step = 0;
    auto best_params = snapshot_parameters(model);
    auto prev_params = best_params;
    std::string stop_reason = "steps";
    int steps_done = 0;
    double last_ess = 1.0;

    // Regione di fiducia con passo adattivo (vedi la versione a un sistema):
    // la soglia vale per il sistema con l'ESS piu' bassa.
    double current_lr = lr;
    int backtracks = 0;
    auto set_lr = [&](double value) {
        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(value);
        }
    };
    int updates = 0;
    int last_logged = -1;
    double prev_ess = 1.0;
    double last_step_lr = -1.0;
    double accepted_lr = -1.0;
    std::cout << "[INFO] passo      ESS/N      dS_train      dS_hold     beta*gap        lr\n";
    while (updates < steps) {
        auto u_cg = current_cg();
        std::vector<torch::Tensor> w_cg;
        double ess = 1.0;
        for (std::size_t k = 0; k < systems.size(); ++k) {
            w_cg.push_back(torch::softmax(-systems[k].beta * (u_cg[k] - systems[k].u0_cg), 0));
            ess = std::min(ess, min_ess_fraction(w_cg.back()));
        }
        last_ess = ess;
        if (ess < ess_min) {
            restore_parameters(model, prev_params);
            --updates;
            if (backtracks >= max_backtracks || prev_ess < 1.2 * ess_min) {
                stop_reason = "ess";
                std::cout << "[INFO] ESS/N = " << ess << " < " << ess_min << " anche con lr "
                          << current_lr << ": serve una nuova simulazione\n";
                break;
            }
            ++backtracks;
            current_lr *= 0.5;
            set_lr(current_lr);
            std::cout << "[INFO] ESS/N = " << ess << " < " << ess_min << " dopo il passo "
                      << (updates + 1) << ": annullato, lr -> " << current_lr << "\n";
            continue;
        }
        if (updates > 0 && last_step_lr > 0.0) accepted_lr = last_step_lr;
        if (updates % eval_every == 0 && updates != last_logged) {
            last_logged = updates;
            json per_system = json::array();
            const Eval e = evaluate(u_cg, &per_system);
            const double score = has_hold ? e.ds_hold : e.ds_train;
            std::cout << "[RELENT] " << std::setw(5) << updates << std::fixed
                      << std::setprecision(4) << std::setw(10) << e.ess
                      << std::setw(14) << e.ds_train << std::setw(13) << e.ds_hold
                      << std::setw(13) << e.gap << std::defaultfloat
                      << std::setw(10) << current_lr << "\n";
            if (systems.size() > 1) {
                for (const auto& ps : per_system) {
                    std::cout << "[RELENT]   T = " << ps["T_K"].get<double>() << " K: ESS/N "
                              << ps["ess"].get<double>() << ", dS_train " << ps["dS_train"].get<double>()
                              << ", dS_hold " << ps["dS_hold"].get<double>() << "\n";
                }
            }
            history.push_back({{"step", updates}, {"ess", e.ess}, {"dS_train", e.ds_train},
                               {"dS_hold", e.ds_hold}, {"beta_gap", e.gap}, {"lr", current_lr},
                               {"systems", per_system}});
            if (score < best_score) {
                best_score = score;
                best_step = updates;
                best_params = snapshot_parameters(model);
            }
        }
        prev_params = snapshot_parameters(model);
        prev_ess = ess;

        optimizer.zero_grad();
        for (std::size_t k = 0; k < systems.size(); ++k) {
            System& s = systems[k];
            auto ia = on_dev(s.aa_train_idx.index_select(
                0, torch::randint(s.n_train, {batch_aa}, torch::kInt64)));
            torch::Tensor loss_cg;
            if (s.cg.groups == 1) {
                auto ic = on_dev(torch::multinomial(w_cg[k].select(1, 0).to(torch::kFloat64), batch_cg, true));
                loss_cg = energy_g(s, s.cg, ic).sum(1).mean();
            } else {
                auto pick = torch::randint(s.cg.frames, {batch_cg}, torch::kInt64);
                auto w_sel = w_cg[k].index_select(0, pick).to(dtype).to(device) *
                             (static_cast<double>(s.cg.frames) / batch_cg);
                loss_cg = (w_sel * energy_g(s, s.cg, on_dev(pick))).sum();
            }
            auto loss = s.weight * s.beta * (energy_g(s, s.aa, ia).sum(1).mean() - loss_cg);
            loss.backward();
        }
        if (grad_clip > 0.0) torch::nn::utils::clip_grad_norm_(train_params, grad_clip);
        optimizer.step();
        last_step_lr = current_lr;
        ++updates;
    }
    steps_done = std::max(0, updates);
    {
        auto u_cg = current_cg();
        json per_system = json::array();
        const Eval e = evaluate(u_cg, &per_system);
        last_ess = e.ess;
        if (e.ess >= ess_min) {
            const double score = has_hold ? e.ds_hold : e.ds_train;
            history.push_back({{"step", steps_done}, {"ess", e.ess}, {"dS_train", e.ds_train},
                               {"dS_hold", e.ds_hold}, {"beta_gap", e.gap}, {"final", true},
                               {"systems", per_system}});
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
              << (has_hold ? "holdout" : "train") << " = " << best_score
              << "), arresto: " << stop_reason << "\n";
    if (best_step == 0) {
        std::cout << "[WARNING] nessun passo abbassa S_rel: il modello resta theta_0\n";
    }
    if (model->has_thermo_heads()) model->set_temperature(model->thermo_T0);
    save_model();
    report["steps_done"] = steps_done;
    report["stop_reason"] = stop_reason;
    report["ess_last"] = last_ess;
    report["lr_final"] = accepted_lr > 0.0 ? accepted_lr : current_lr;
    report["backtracks"] = backtracks;
    report["best_step"] = best_step;
    report["best_dS"] = best_score;
    report["best_dS_source"] = has_hold ? "holdout" : "train";
    json frames = json::array();
    for (const auto& s : systems) {
        frames.push_back({{"T_K", s.T}, {"aa_train", s.n_train}, {"aa_holdout", s.n_hold},
                          {"aa_monitor", s.aa_mon_idx.size(0)}, {"cg", s.cg.frames}});
    }
    report["frames"] = systems.size() == 1 ? frames[0] : frames;
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
