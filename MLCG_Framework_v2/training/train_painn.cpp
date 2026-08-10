#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <limits>
#include <random>    // Aggiungi questo in cima al file per std::shuffle
#include <algorithm> // Aggiungi questo in cima per std::shuffle
#include <stdexcept>
#include <filesystem>
#include <cmath>

#include "json.hpp"
using json = nlohmann::json;

#ifdef __APPLE__
#include <ATen/mps/MPSAllocatorInterface.h>
#endif

// REMARK: a molecule (molecule_id) here is a real particle complemented with its virtual sites

// Assicurati di avere questo file nella stessa cartella
#include "PaiNN_Architecture.hpp"

// =====================================================================
// 1. STRUTTURE DATI (Siti, Molecole, Frame)
// =====================================================================
struct CGSite {
    int molecule_id; 
    int site_type;   // Usato come "atomic_number" per l'embedding di PaiNN
    float x, y, z;
};

struct CGMolecule {
    int molecule_id;
    std::vector<CGSite> sites;
    
    float center_of_geometry[3]; // Punto di riferimento per il momento torcente (es. posizione particella reale)
    float target_force[3];       // Forza totale target (Ground Truth)
    float target_torque[3];      // Momento torcente target (Ground Truth)
};

struct CGFrame {
    std::vector<CGMolecule> molecules;
    float box[3];
};

// Batch di tensori pronti per la GPU
struct CGBatch {
    torch::Tensor site_types;    // [N_sites]
    torch::Tensor coordinates;   // [N_sites, 3]
    torch::Tensor edge_index;    // [2, N_edges]
    torch::Tensor batch_indices; // [N_sites] (Mappatura sito -> Frame nel batch)
    
    torch::Tensor mol_indices;   // [N_sites] (Mappatura sito -> Molecola nel batch)
    torch::Tensor mol_centers;   // [N_mols, 3]
    
    torch::Tensor target_mol_forces;  // [N_mols, 3]
    torch::Tensor target_mol_torques; // [N_mols, 3]
    
    torch::Tensor frame_boxes;        // [N_frames, 3]
    
    int num_molecules_in_batch;
};

// =====================================================================
// 2. FUNZIONI DI SUPPORTO E BATCHING
// =====================================================================
CGBatch collate_batch(const std::vector<CGFrame>& frames, float cutoff, torch::Device device) {
    CGBatch batch;
    std::vector<int64_t> site_types_vec, batch_indices_vec, mol_indices_vec;
    std::vector<float> coords_vec, centers_vec, forces_vec, torques_vec;
    std::vector<int64_t> edge_rows, edge_cols;
    
    std::vector<float> frame_boxes_vec;
    frame_boxes_vec.reserve(frames.size() * 3);
    
    // Esempio per evitare riallocazioni
    size_t estimated_sites = frames.size() * frames[0].molecules.size();
    site_types_vec.reserve(estimated_sites);
    coords_vec.reserve(estimated_sites * 3);
    int site_offset = 0;
    int global_mol_idx = 0;
    float cutoff_sq = cutoff * cutoff;

    for (size_t b_idx = 0; b_idx < frames.size(); ++b_idx) {
        const auto& frame = frames[b_idx];
        frame_boxes_vec.push_back(frame.box[0]);
        frame_boxes_vec.push_back(frame.box[1]);
        frame_boxes_vec.push_back(frame.box[2]);
        float box_x = frame.box[0];
        float box_y = frame.box[1];
        float box_z = frame.box[2];
        
        int frame_site_start = site_offset;
        
        std::vector<CGSite> frame_sites; // Flatten dei siti per calcolare le distanze
        
        for (const auto& mol : frame.molecules) {
            centers_vec.insert(centers_vec.end(), {mol.center_of_geometry[0], mol.center_of_geometry[1], mol.center_of_geometry[2]});
            forces_vec.insert(forces_vec.end(), {mol.target_force[0], mol.target_force[1], mol.target_force[2]});
            torques_vec.insert(torques_vec.end(), {mol.target_torque[0], mol.target_torque[1], mol.target_torque[2]});
            
            for (const auto& site : mol.sites) {
                site_types_vec.push_back(site.site_type);
                coords_vec.insert(coords_vec.end(), {site.x, site.y, site.z});
                batch_indices_vec.push_back(b_idx);
                mol_indices_vec.push_back(global_mol_idx);
                frame_sites.push_back(site);
            }
            global_mol_idx++;
        }
        
        // COSTRUZIONE DEL GRAFO CON CONDIZIONI PERIODICHE (PBC)
        int num_sites_in_frame = frame_sites.size();
        for (int i = 0; i < num_sites_in_frame; ++i) {
            for (int j = i + 1; j < num_sites_in_frame; ++j) {
                // REGOLA: Se appartengono alla stessa molecola, NON interagiscono
                if (frame_sites[i].molecule_id == frame_sites[j].molecule_id) continue;
                
                // 1. Distanza cartesiana iniziale
                float dx = frame_sites[i].x - frame_sites[j].x;
                float dy = frame_sites[i].y - frame_sites[j].y;
                float dz = frame_sites[i].z - frame_sites[j].z;
                
                // 2. Applicazione della Minimum Image Convention per le PBC
                dx -= box_x * std::round(dx / box_x);
                dy -= box_y * std::round(dy / box_y);
                dz -= box_z * std::round(dz / box_z);

                // 3. Controllo del cutoff sulla distanza reale periodica
                if ((dx*dx + dy*dy + dz*dz) <= cutoff_sq) {
                    edge_rows.push_back(frame_site_start + i); edge_cols.push_back(frame_site_start + j);
                    edge_rows.push_back(frame_site_start + j); edge_cols.push_back(frame_site_start + i);
                }
            }
        }
        site_offset += num_sites_in_frame;
    }
    
    // Creazione Tensori PyTorch
    batch.frame_boxes = torch::tensor(frame_boxes_vec, torch::kFloat32).reshape({-1, 3}).to(device);
    batch.site_types = torch::tensor(site_types_vec, torch::kInt64).to(device);
    batch.coordinates = torch::tensor(coords_vec, torch::kFloat32).reshape({-1, 3}).to(device);
    batch.batch_indices = torch::tensor(batch_indices_vec, torch::kInt64).to(device);
    batch.mol_indices = torch::tensor(mol_indices_vec, torch::kInt64).to(device);
    
    batch.mol_centers = torch::tensor(centers_vec, torch::kFloat32).reshape({-1, 3}).to(device);
    batch.target_mol_forces = torch::tensor(forces_vec, torch::kFloat32).reshape({-1, 3}).to(device);
    batch.target_mol_torques = torch::tensor(torques_vec, torch::kFloat32).reshape({-1, 3}).to(device);
    
    batch.num_molecules_in_batch = global_mol_idx;
    
    if (!edge_rows.empty()) {
        std::vector<int64_t> flat_edges;
        flat_edges.insert(flat_edges.end(), edge_rows.begin(), edge_rows.end());
        flat_edges.insert(flat_edges.end(), edge_cols.begin(), edge_cols.end());
        batch.edge_index = torch::tensor(flat_edges, torch::kInt64).reshape({2, (long)edge_rows.size()}).to(device);
    } else {
        batch.edge_index = torch::empty({2, 0}, torch::dtype(torch::kInt64).device(device));
    }
    
    return batch;
}
static void read_exact(std::ifstream& file, void* dst, std::size_t nbytes, const std::string& what) {
    file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(nbytes));
    if (!file) {
        throw std::runtime_error("Truncated/corrupt CG dataset while reading " + what);
    }
}

static bool finite3(const float values[3]) {
    return std::isfinite(values[0]) && std::isfinite(values[1]) && std::isfinite(values[2]);
}

std::vector<CGFrame> read_cg_dataset(const std::string& filepath, int num_species) {
    std::vector<CGFrame> dataset;
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Impossibile aprire il dataset binario: " + filepath);
    }

    int num_frames = 0;
    read_exact(file, &num_frames, sizeof(int), "num_frames");
    if (num_frames <= 0 || num_frames > 10000000) {
        throw std::runtime_error("Invalid num_frames in CG dataset: " + std::to_string(num_frames));
    }
    dataset.reserve(static_cast<std::size_t>(num_frames));

    for (int f = 0; f < num_frames; ++f) {
        CGFrame frame;
        int num_molecules = 0;
        int num_total_sites = 0;

        read_exact(file, &num_molecules, sizeof(int), "frame num_molecules");
        read_exact(file, &num_total_sites, sizeof(int), "frame num_total_sites");
        read_exact(file, frame.box, 3 * sizeof(float), "frame box");
        if (num_molecules <= 0 || num_molecules > 10000000) {
            throw std::runtime_error("Invalid num_molecules at frame " + std::to_string(f));
        }
        if (num_total_sites < num_molecules || num_total_sites > 100000000) {
            throw std::runtime_error("Invalid num_total_sites at frame " + std::to_string(f));
        }
        if (!finite3(frame.box) || frame.box[0] <= 0.0f || frame.box[1] <= 0.0f || frame.box[2] <= 0.0f) {
            throw std::runtime_error("Invalid/non-finite box at frame " + std::to_string(f));
        }

        frame.molecules.reserve(static_cast<std::size_t>(num_molecules));
        int counted_sites = 0;

        for (int m = 0; m < num_molecules; ++m) {
            CGMolecule mol;
            int num_sites = 0;

            read_exact(file, &mol.molecule_id, sizeof(int), "molecule_id");
            read_exact(file, &num_sites, sizeof(int), "num_sites");
            if (mol.molecule_id != m) {
                throw std::runtime_error(
                    "Non-sequential molecule_id at frame " + std::to_string(f) +
                    ": expected " + std::to_string(m) + ", got " + std::to_string(mol.molecule_id));
            }
            if (num_sites <= 0 || num_sites > num_total_sites) {
                throw std::runtime_error("Invalid num_sites at frame " + std::to_string(f) +
                                         ", molecule " + std::to_string(m));
            }

            read_exact(file, mol.center_of_geometry, 3 * sizeof(float), "molecule center");
            read_exact(file, mol.target_force, 3 * sizeof(float), "molecule target_force");
            read_exact(file, mol.target_torque, 3 * sizeof(float), "molecule target_torque");
            if (!finite3(mol.center_of_geometry) || !finite3(mol.target_force) || !finite3(mol.target_torque)) {
                throw std::runtime_error("Non-finite molecule data at frame " + std::to_string(f) +
                                         ", molecule " + std::to_string(m));
            }

            mol.sites.reserve(static_cast<std::size_t>(num_sites));
            for (int site_idx = 0; site_idx < num_sites; ++site_idx) {
                CGSite site;
                site.molecule_id = mol.molecule_id;
                read_exact(file, &site.site_type, sizeof(int), "site_type");
                read_exact(file, &site.x, sizeof(float), "site x");
                read_exact(file, &site.y, sizeof(float), "site y");
                read_exact(file, &site.z, sizeof(float), "site z");
                if (site.site_type < 0 || site.site_type >= num_species) {
                    throw std::runtime_error(
                        "site_type out of range at frame " + std::to_string(f) +
                        ", molecule " + std::to_string(m) +
                        ": " + std::to_string(site.site_type));
                }
                if (!std::isfinite(site.x) || !std::isfinite(site.y) || !std::isfinite(site.z)) {
                    throw std::runtime_error("Non-finite site coordinate at frame " + std::to_string(f) +
                                             ", molecule " + std::to_string(m));
                }
                mol.sites.push_back(site);
            }
            counted_sites += num_sites;
            frame.molecules.push_back(std::move(mol));
        }

        if (counted_sites != num_total_sites) {
            throw std::runtime_error(
                "num_total_sites mismatch at frame " + std::to_string(f) +
                ": header=" + std::to_string(num_total_sites) +
                ", parsed=" + std::to_string(counted_sites));
        }
        dataset.push_back(std::move(frame));
    }

    char trailing = 0;
    if (file.read(&trailing, 1)) {
        throw std::runtime_error("CG dataset contains trailing bytes after declared frames");
    }
    if (!file.eof()) {
        throw std::runtime_error("I/O error while checking end of CG dataset");
    }

    std::cout << "[INFO] Letti " << dataset.size() << " frame dal dataset.\n";
    return dataset;
}

// Legacy OOD decoys generated by older build_cg_dataset.py versions carry an
// exactly zero residual force and torque target for every molecule.  We still
// detect them so existing datasets can be audited and safely exclude them from
// optimization unless an explicit legacy-ablation flag is set.
bool is_zero_target_decoy_frame(const CGFrame& frame) {
    if (frame.molecules.empty()) return false;
    for (const auto& mol : frame.molecules) {
        for (int k = 0; k < 3; ++k) {
            if (mol.target_force[k] != 0.0f || mol.target_torque[k] != 0.0f) {
                return false;
            }
        }
    }
    return true;
}
// =====================================================================
// 3. LOGICA DI TRAINING (Metrice ed Early Stopping)
// =====================================================================
struct Metrics {
    float loss;
    float mae_forces;
    float mae_torques;
};

struct EarlyStopping {
    int patience, counter;
    float best_loss;
    bool early_stop;
    std::string save_path;

    EarlyStopping(int p, std::string path) : 
        patience(p), counter(0), best_loss(std::numeric_limits<float>::infinity()), 
        early_stop(false), save_path(path) {}

    void check(PaiNNModel& model, float val_loss, torch::Device device) { 
        if (val_loss < best_loss) {
            best_loss = val_loss;
            counter = 0;
            model->to(torch::kCPU);
            torch::save(model, save_path);
            model->to(device);
            std::cout << "   ---> [Early Stopping] Miglioramento! Modello salvato.\n";
        } else {
            counter++;
            std::cout << "   ---> [Early Stopping] Nessun miglioramento (" << counter << "/" << patience << ").\n";
            if (counter >= patience) early_stop = true;
        }
    }
};

void progress_bar(double progresso) 
{
    int len_barra = 50; // Lunghezza totale della barra in caratteri
    std::cout << "[";

    int pos = len_barra * progresso;
    std::cout << "\033[32m"; // colore verde
    for (int i = 0; i < len_barra; ++i) 
    {
        if (i < pos) 
            std::cout << "=";
        else if (i == pos) 
            std::cout << ">";
        else 
            std::cout << " ";
    }
    std::cout << "\033[0m"; // reset colore 

    // Mostra la percentuale e usa \r per tornare all'inizio della riga
    std::cout << "] " << int(progresso * 100.0) << " %\r";
    std::cout.flush(); // Forza l'output immediato sul terminale
    if (pos == len_barra)
        std::cout << "\n";
}
 
static std::uintmax_t file_size_or_zero(const std::string& path) {
    std::error_code ec;
    auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : size;
}

static void validate_resume_manifest(
    const std::string& model_path,
    const std::string& dataset_path,
    const std::string& config_path,
    const json& effective_config) {
    const std::string manifest_path = model_path + ".manifest.json";
    std::ifstream input(manifest_path);
    if (!input.is_open()) {
        throw std::runtime_error(
            "Existing model has no manifest: " + manifest_path +
            ". Delete/rename the model to start fresh, or create a valid manifest first.");
    }
    json manifest;
    input >> manifest;
    if (manifest.value("schema_version", -1) != 3 ||
        manifest.value("framework", std::string()) != "MLCG_Framework_v2") {
        throw std::runtime_error("Unsupported model manifest: " + manifest_path);
    }
    if (manifest.value("energy_gauge", std::string()) != "isolated_species_zero_v1") {
        throw std::runtime_error(
            "Cannot resume: unsupported or missing energy gauge in " + manifest_path);
    }
    const auto& architecture = manifest.at("architecture");
    if (architecture.value("variant", std::string()) !=
        effective_config.at("architecture_variant").get<std::string>()) {
        throw std::runtime_error("Cannot resume: model manifest mismatch for architecture variant");
    }
    const std::vector<std::string> integer_keys = {
        "num_species", "hidden_channels", "n_layers", "num_rbf"};
    for (const auto& key : integer_keys) {
        if (architecture.at(key).get<int>() != effective_config.at(key).get<int>()) {
            throw std::runtime_error("Cannot resume: model manifest mismatch for " + key);
        }
    }
    for (const auto& key : {std::string("cutoff"), std::string("toxvaerd_alpha")}) {
        if (std::abs(architecture.at(key).get<double>() -
                     effective_config.at(key).get<double>()) > 1e-12) {
            throw std::runtime_error("Cannot resume: model manifest mismatch for " + key);
        }
    }
    if (manifest.contains("model_file_size_bytes") &&
        manifest.at("model_file_size_bytes").get<std::uintmax_t>() != file_size_or_zero(model_path)) {
        throw std::runtime_error("Cannot resume: model size differs from its manifest");
    }
    if (manifest.contains("dataset_file_size_bytes") &&
        manifest.at("dataset_file_size_bytes").get<std::uintmax_t>() != file_size_or_zero(dataset_path)) {
        throw std::runtime_error("Cannot resume: dataset size differs from the model manifest");
    }
    if (manifest.contains("config_file_size_bytes") &&
        manifest.at("config_file_size_bytes").get<std::uintmax_t>() != file_size_or_zero(config_path)) {
        throw std::runtime_error("Cannot resume: config size differs from the model manifest");
    }
}

static void write_model_manifest(
    const std::string& model_path,
    const std::string& dataset_path,
    const std::string& config_path,
    const json& effective_config,
    float best_validation_loss) {
    json architecture = {
        {"variant", effective_config.at("architecture_variant")},
        {"num_species", effective_config.at("num_species")},
        {"hidden_channels", effective_config.at("hidden_channels")},
        {"n_layers", effective_config.at("n_layers")},
        {"num_rbf", effective_config.at("num_rbf")},
        {"cutoff", effective_config.at("cutoff")},
        {"toxvaerd_alpha", effective_config.at("toxvaerd_alpha")},
    };
    json manifest = {
        {"schema_version", 3},
        {"framework", "MLCG_Framework_v2"},
        {"energy_gauge", "isolated_species_zero_v1"},
        {"architecture", architecture},
        {"effective_config", effective_config},
        {"model_path", model_path},
        {"model_file_size_bytes", file_size_or_zero(model_path)},
        {"dataset_path", dataset_path},
        {"dataset_file_size_bytes", file_size_or_zero(dataset_path)},
        {"config_path", config_path},
        {"config_file_size_bytes", file_size_or_zero(config_path)},
        {"split_seed", effective_config.at("split_seed")},
        {"validation_fraction", effective_config.at("validation_fraction")},
        {"physical_validation_only", effective_config.at("physical_validation_only")},
        {"best_validation_loss", best_validation_loss},
        {"force_units", "kJ mol^-1 nm^-1"},
        {"torque_units", "kJ mol^-1"},
    };

    const std::string manifest_path = model_path + ".manifest.json";
    std::ofstream output(manifest_path);
    if (!output.is_open()) {
        throw std::runtime_error("Cannot write model manifest: " + manifest_path);
    }
    output << manifest.dump(2) << "\n";
    std::cout << "[INFO] Model manifest written to: " << manifest_path << "\n";
}

#ifdef __APPLE__
extern "C" {
    void* objc_autoreleasePoolPush(void);
    void objc_autoreleasePoolPop(void* pool);
}
#endif

int main(int argc, char* argv[]) {
    torch::manual_seed(42);
    
    // FIX: Utilizziamo torch::Device in modo pulito ed esplicito
    torch::Device device(torch::kCPU);
    std::string device_name = "CPU";

    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        device_name = "CUDA";
    }
    else if (torch::mps::is_available()) {
        device = torch::Device(torch::kMPS);
        device_name = "MPS (GPU Mac)";
    }

    std::cout << "[INFO] Utilizzando il device: " << device_name << "\n";
    
    // 1. Parametri Rete (letti da JSON)
    int num_species = 100; 
    int dim = 128;
    int layers = 3;
    int num_rbf = 40; 
    float cutoff = 5.0f;
    int max_epochs = 500;
    float initial_lr = 5e-4;
    float lipschitz_lambda = 0.0f;
    float weight_decay_val = 0.0f;
    int es_patience = 10;
    int reduce_lr_patience = 5;
    int batch_size = 16;
    int diagnostic_overfit_frames = 0;
    bool physical_validation_only = true;
    bool include_decoys_in_train = false;
    bool shuffle_each_epoch = true;
    int split_seed = 42;
    float validation_fraction = 0.2f;

    std::string dataset_path = "cg_dataset.bin";
    std::string model_path = "best_cg_model.pt";
    std::string config_path = "cg_model_config.json";
    bool resume_training = false;
    
    if (argc >= 2) dataset_path = argv[1];
    if (argc >= 3) model_path = argv[2];
    if (argc >= 4) config_path = argv[3];
    for (int i = 4; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--resume") {
            resume_training = true;
        } else {
            std::cerr << "[ERROR] Unknown training option: " << option << "\n";
            return 2;
        }
    }


    double toxvaerd_alpha = 0.1;
    float torque_weight = 0.0f;
    float grad_clip_norm = 1.0f;
    json loaded_config = json::object();

    // Lettura JSON
    std::ifstream config_file(config_path);
    if (config_file.is_open()) {
        config_file >> loaded_config;
        if (loaded_config.contains("num_species")) num_species = loaded_config["num_species"];
        if (loaded_config.contains("hidden_channels")) dim = loaded_config["hidden_channels"];
        if (loaded_config.contains("n_layers")) layers = loaded_config["n_layers"];
        if (loaded_config.contains("num_rbf")) num_rbf = loaded_config["num_rbf"];
        if (loaded_config.contains("cutoff")) cutoff = loaded_config["cutoff"];

        if (loaded_config.contains("toxvaerd_alpha")) toxvaerd_alpha = loaded_config["toxvaerd_alpha"];
        if (loaded_config.contains("epochs")) max_epochs = loaded_config["epochs"];
        if (loaded_config.contains("learning_rate")) initial_lr = loaded_config["learning_rate"];
        if (loaded_config.contains("weight_decay")) weight_decay_val = loaded_config["weight_decay"];
        if (loaded_config.contains("lipschitz_lambda")) lipschitz_lambda = loaded_config["lipschitz_lambda"];
        if (loaded_config.contains("early_stopping_patience")) es_patience = loaded_config["early_stopping_patience"];
        if (loaded_config.contains("reduce_lr_patience")) reduce_lr_patience = loaded_config["reduce_lr_patience"];
        if (loaded_config.contains("torque_weight")) torque_weight = loaded_config["torque_weight"];
        if (loaded_config.contains("grad_clip_norm")) grad_clip_norm = loaded_config["grad_clip_norm"];
        if (loaded_config.contains("batch_size")) batch_size = loaded_config["batch_size"];
        if (loaded_config.contains("diagnostic_overfit_frames")) diagnostic_overfit_frames = loaded_config["diagnostic_overfit_frames"];
        if (loaded_config.contains("physical_validation_only")) physical_validation_only = loaded_config["physical_validation_only"];
        if (loaded_config.contains("include_decoys_in_train")) include_decoys_in_train = loaded_config["include_decoys_in_train"];
        if (loaded_config.contains("shuffle_each_epoch")) shuffle_each_epoch = loaded_config["shuffle_each_epoch"];
        if (loaded_config.contains("split_seed")) split_seed = loaded_config["split_seed"];
        if (loaded_config.contains("validation_fraction")) validation_fraction = loaded_config["validation_fraction"];
        if (batch_size <= 0) {
            throw std::runtime_error("batch_size must be positive");
        }
        if (diagnostic_overfit_frames < 0) {
            throw std::runtime_error("diagnostic_overfit_frames must be non-negative (0 disables tiny-set mode)");
        }
        if (validation_fraction <= 0.0f || validation_fraction >= 1.0f) {
            throw std::runtime_error("validation_fraction must be strictly between 0 and 1");
        }
        if (torque_weight < 0.0f) {
            throw std::runtime_error("torque_weight must be non-negative");
        }
        if (grad_clip_norm < 0.0f) {
            throw std::runtime_error("grad_clip_norm must be non-negative (0 disables clipping)");
        }
        std::cout << "[INFO] Caricati iperparametri da " << config_path << "\n";
    } else {
        std::cerr << "[ERROR] Cannot read required training config: " << config_path << "\n";
        return 2;
    }

    const std::string expected_architecture_variant(PAINN_ARCHITECTURE_VARIANT);
    const std::string configured_architecture_variant =
        loaded_config.value("architecture_variant", std::string());
    if (configured_architecture_variant != expected_architecture_variant) {
        throw std::runtime_error(
            "Training config architecture_variant must be '" + expected_architecture_variant +
            "' for this build; got '" + configured_architecture_variant + "'");
    }

    json effective_config = loaded_config;
    effective_config["architecture_variant"] = expected_architecture_variant;
    effective_config["num_species"] = num_species;
    effective_config["hidden_channels"] = dim;
    effective_config["n_layers"] = layers;
    effective_config["num_rbf"] = num_rbf;
    effective_config["cutoff"] = cutoff;
    effective_config["toxvaerd_alpha"] = toxvaerd_alpha;
    effective_config["epochs"] = max_epochs;
    effective_config["learning_rate"] = initial_lr;
    effective_config["weight_decay"] = weight_decay_val;
    effective_config["lipschitz_lambda"] = lipschitz_lambda;
    effective_config["early_stopping_patience"] = es_patience;
    effective_config["reduce_lr_patience"] = reduce_lr_patience;
    effective_config["torque_weight"] = torque_weight;
    effective_config["grad_clip_norm"] = grad_clip_norm;
    effective_config["batch_size"] = batch_size;
    effective_config["diagnostic_overfit_frames"] = diagnostic_overfit_frames;
    effective_config["physical_validation_only"] = physical_validation_only;
    effective_config["include_decoys_in_train"] = include_decoys_in_train;
    effective_config["shuffle_each_epoch"] = shuffle_each_epoch;
    effective_config["split_seed"] = split_seed;
    effective_config["validation_fraction"] = validation_fraction;
    effective_config["decoy_detection"] = "exact_zero_residual_target_v1";
    effective_config["loss_normalization"] = "train_target_rms_v1";
    effective_config["energy_gauge"] = "isolated_species_zero_v1";

    // Inizializza il Modello
    PaiNNModel model(num_species, dim, layers, num_rbf, cutoff, toxvaerd_alpha);
    std::ifstream f(model_path.c_str());
    if (f.good()) {
        if (!resume_training) {
            std::cerr << "[ERROR] Output model already exists: " << model_path
                      << ". Refusing an implicit resume. Use a new output path, delete the old "
                         "artifact, or pass --resume deliberately.\n";
            return 2;
        }
        try {
            validate_resume_manifest(model_path, dataset_path, config_path, effective_config);
            torch::load(model, model_path);
            std::cout << "[INFO] Modello esistente e manifest coerente caricati da: "
                      << model_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Existing model cannot be resumed safely: "
                      << e.what() << "\n";
            return 2;
        }
    } else {
        if (resume_training) {
            std::cerr << "[ERROR] --resume requested but model does not exist: "
                      << model_path << "\n";
            return 2;
        }
        std::cout << "[INFO] Nessun modello esistente trovato in " << model_path
                  << ". Inizio training da zero.\n";
    }
    model->to(device);

    // Iperparametri Training 
    float current_lr = initial_lr; 
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(initial_lr).weight_decay(weight_decay_val));
    EarlyStopping early_stopping(es_patience, model_path);
    
    int lr_patience = reduce_lr_patience; 
    int lr_counter = 0;
    float best_val_loss = std::numeric_limits<float>::max();
    
    std::ofstream csv_file("cg_training_log.csv");
    if (csv_file.is_open()) {
        csv_file << "Epoch,Train_Loss,Val_Loss,Train_Loss_F_Norm,Train_Loss_T_Norm,"
                    "Val_Loss_F_Norm,Val_Loss_T_Norm,Train_MAE_F,Train_MAE_T,"
                    "Val_MAE_F,Val_MAE_T,Val_Zero_F_Norm,Val_Zero_T_Norm,Val_Zero_Total\n";
    }

    std::cout << "\n[INFO] Caricamento dataset binario in corso: " << dataset_path << "...\n";
    
    std::vector<CGFrame> full_dataset = read_cg_dataset(dataset_path, num_species);
    
    if (full_dataset.empty()) {
        std::cerr << "Errore critico: dataset vuoto o file non trovato. Interruzione.\n";
        return 1;
    }

    std::mt19937 g(static_cast<std::mt19937::result_type>(split_seed));

    std::vector<CGFrame> physical_frames;
    std::vector<CGFrame> decoy_frames;
    physical_frames.reserve(full_dataset.size());
    decoy_frames.reserve(full_dataset.size());
    for (auto& frame : full_dataset) {
        if (is_zero_target_decoy_frame(frame)) {
            decoy_frames.push_back(std::move(frame));
        } else {
            physical_frames.push_back(std::move(frame));
        }
    }
    const size_t detected_physical_frames = physical_frames.size();
    const size_t detected_decoy_frames = decoy_frames.size();

    if (physical_frames.empty()) {
        throw std::runtime_error(
            "No physical (non-zero-target) frames detected in the CG dataset");
    }
    if (detected_decoy_frames > 0 && include_decoys_in_train) {
        std::cerr
            << "[WARNING] include_decoys_in_train=true enables the legacy whole-frame "
               "zero-target decoys. These frames do not carry per-molecule loss masks and "
               "can impose artificial zero labels on molecules outside the OOD contact. "
               "Use only for an explicit legacy ablation.\n";
    }

    std::vector<CGFrame> train_dataset;
    std::vector<CGFrame> val_dataset;

    if (diagnostic_overfit_frames > 0) {
        // Tiny-set diagnostics should use physical targets by default.  Legacy
        // zero-target decoys are included only through the explicit unsafe flag.
        std::vector<CGFrame> diagnostic_pool = physical_frames;
        if (include_decoys_in_train) {
            diagnostic_pool.insert(
                diagnostic_pool.end(), decoy_frames.begin(), decoy_frames.end());
        }
        std::shuffle(diagnostic_pool.begin(), diagnostic_pool.end(), g);
        const size_t n = std::min(
            static_cast<size_t>(diagnostic_overfit_frames), diagnostic_pool.size());
        if (n == 0) {
            throw std::runtime_error("Tiny-set diagnostic requested but dataset is empty");
        }
        train_dataset.assign(diagnostic_pool.begin(), diagnostic_pool.begin() + n);
        val_dataset = train_dataset;
        std::cout << "[DIAGNOSTIC] Tiny-set overfit mode enabled: " << n
                  << " deterministic frames are used for BOTH train and validation.\n"
                  << "             Do not interpret this validation loss as generalization.\n";
    } else if (physical_validation_only) {
        if (physical_frames.size() < 2) {
            throw std::runtime_error(
                "physical_validation_only requested but fewer than two physical frames were detected");
        }

        // The physical split is deterministic and independent of cutoff/model settings.
        std::shuffle(physical_frames.begin(), physical_frames.end(), g);
        size_t val_size = static_cast<size_t>(
            static_cast<double>(physical_frames.size()) * validation_fraction);
        val_size = std::max<size_t>(1, std::min(val_size, physical_frames.size() - 1));
        const size_t train_physical_size = physical_frames.size() - val_size;

        train_dataset.assign(
            physical_frames.begin(), physical_frames.begin() + train_physical_size);
        val_dataset.assign(
            physical_frames.begin() + train_physical_size, physical_frames.end());
        if (include_decoys_in_train) {
            std::mt19937 g_decoy(static_cast<std::mt19937::result_type>(split_seed + 1));
            std::shuffle(decoy_frames.begin(), decoy_frames.end(), g_decoy);
            train_dataset.insert(train_dataset.end(), decoy_frames.begin(), decoy_frames.end());
        }

        std::cout << "[INFO] Physical-only validation enabled.\n"
                  << "       - Detected physical frames: " << detected_physical_frames << "\n"
                  << "       - Detected zero-target OOD decoys: " << detected_decoy_frames << "\n"
                  << "       - Physical train: " << train_physical_size << "\n"
                  << "       - Decoys included in train: "
                  << (include_decoys_in_train ? detected_decoy_frames : 0) << "\n"
                  << "       - Decoys excluded from optimization: "
                  << (include_decoys_in_train ? 0 : detected_decoy_frames) << "\n"
                  << "       - Physical validation: " << val_dataset.size() << "\n"
                  << "       - Split seed: " << split_seed
                  << " | validation_fraction=" << validation_fraction << "\n";
    } else {
        // Generic split.  Safe default: discard legacy unmasked decoys rather
        // than mixing artificial zero-target frames into train/validation.
        std::vector<CGFrame> split_pool = physical_frames;
        if (include_decoys_in_train) {
            split_pool.insert(split_pool.end(), decoy_frames.begin(), decoy_frames.end());
        }
        std::shuffle(split_pool.begin(), split_pool.end(), g);
        if (split_pool.size() < 2) {
            throw std::runtime_error("Need at least two frames for train/validation split");
        }
        size_t val_size = static_cast<size_t>(
            static_cast<double>(split_pool.size()) * validation_fraction);
        val_size = std::max<size_t>(1, std::min(val_size, split_pool.size() - 1));
        const size_t train_size = split_pool.size() - val_size;
        train_dataset.assign(split_pool.begin(), split_pool.begin() + train_size);
        val_dataset.assign(split_pool.begin() + train_size, split_pool.end());
    }

    std::cout << "[INFO] Split completato:\n"
              << "       - Train: " << train_dataset.size() << " frames\n"
              << "       - Val:   " << val_dataset.size() << " frames\n\n";

    // -----------------------------------------------------------------
    // Scale fisiche del train set per una loss adimensionale e bilanciata.
    // Force/torque MAE remain in physical units for interpretable logging.
    // Torque statistics include only multi-site rigid bodies, exactly as the
    // torque loss mask used below.
    // -----------------------------------------------------------------
    double force_sum2 = 0.0;
    double force_abs_sum = 0.0;
    long   force_count = 0;
    double torque_sum2 = 0.0;
    double torque_abs_sum = 0.0;
    long   torque_count = 0;
    long   torque_molecule_count = 0;

    for (const auto& frame : train_dataset) {
        for (const auto& mol : frame.molecules) {
            for (int k = 0; k < 3; ++k) {
                const double f = static_cast<double>(mol.target_force[k]);
                force_sum2 += f * f;
                force_abs_sum += std::abs(f);
                force_count++;
            }
            if (mol.sites.size() > 1) {
                torque_molecule_count++;
                for (int k = 0; k < 3; ++k) {
                    const double t = static_cast<double>(mol.target_torque[k]);
                    torque_sum2 += t * t;
                    torque_abs_sum += std::abs(t);
                    torque_count++;
                }
            }
        }
    }

    float force_rms = (force_count > 0)
        ? static_cast<float>(std::sqrt(force_sum2 / force_count)) : 1.0f;
    float torque_rms = (torque_count > 0)
        ? static_cast<float>(std::sqrt(torque_sum2 / torque_count)) : 1.0f;
    if (force_rms < 1e-6f) force_rms = 1.0f;
    if (torque_rms < 1e-6f) torque_rms = 1.0f;

    const float force_zero_mae = (force_count > 0)
        ? static_cast<float>(force_abs_sum / force_count) : 0.0f;
    const float torque_zero_mae = (torque_count > 0)
        ? static_cast<float>(torque_abs_sum / torque_count) : 0.0f;
    const float force_scale2 = force_rms * force_rms;
    const float torque_scale2 = torque_rms * torque_rms;

    if (torque_weight > 0.0f && torque_count == 0) {
        throw std::runtime_error(
            "torque_weight > 0 but the training split contains no multi-site molecules");
    }

    std::cout << "[INFO] Force RMS (train): " << force_rms
              << " kJ/(mol*nm) | zero-predictor MAE: " << force_zero_mae << "\n";
    if (torque_count > 0) {
        std::cout << "[INFO] Torque RMS (train, multi-site only): " << torque_rms
                  << " kJ/mol | zero-predictor MAE: " << torque_zero_mae
                  << " | samples: " << torque_molecule_count << " molecules\n";
    } else {
        std::cout << "[INFO] Torque RMS: n/a (no multi-site molecules in train split)\n";
    }
    std::cout << "[INFO] Loss normalization: MSE(F)/ForceRMS^2 + "
              << torque_weight << " * MSE(T)/TorqueRMS^2"
              << " | grad_clip_norm=" << grad_clip_norm << "\n";

    // Exact zero-predictor baseline on validation, normalized with TRAIN scales.
    // This is the correct reference for deciding whether validation learns anything.
    double val_force_sum2 = 0.0;
    double val_force_abs_sum = 0.0;
    long val_force_count = 0;
    double val_torque_sum2 = 0.0;
    double val_torque_abs_sum = 0.0;
    long val_torque_count = 0;
    for (const auto& frame : val_dataset) {
        for (const auto& mol : frame.molecules) {
            for (int k = 0; k < 3; ++k) {
                const double fval = static_cast<double>(mol.target_force[k]);
                val_force_sum2 += fval * fval;
                val_force_abs_sum += std::abs(fval);
                val_force_count++;
            }
            if (mol.sites.size() > 1) {
                for (int k = 0; k < 3; ++k) {
                    const double tval = static_cast<double>(mol.target_torque[k]);
                    val_torque_sum2 += tval * tval;
                    val_torque_abs_sum += std::abs(tval);
                    val_torque_count++;
                }
            }
        }
    }
    const float val_zero_f_norm = (val_force_count > 0)
        ? static_cast<float>((val_force_sum2 / val_force_count) / force_scale2) : 0.0f;
    const float val_zero_t_norm = (val_torque_count > 0)
        ? static_cast<float>((val_torque_sum2 / val_torque_count) / torque_scale2) : 0.0f;
    const float val_zero_mae_f = (val_force_count > 0)
        ? static_cast<float>(val_force_abs_sum / val_force_count) : 0.0f;
    const float val_zero_mae_t = (val_torque_count > 0)
        ? static_cast<float>(val_torque_abs_sum / val_torque_count) : 0.0f;
    const float val_zero_total = val_zero_f_norm + torque_weight * val_zero_t_norm;

    std::cout << "[INFO] Validation zero-predictor baseline (TRAIN-normalized): "
              << "F=" << val_zero_f_norm
              << " | T=" << val_zero_t_norm
              << " | weighted total=" << val_zero_total
              << " | MAE F=" << val_zero_mae_f
              << " | MAE T=" << val_zero_mae_t << "\n\n";

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        if (shuffle_each_epoch && train_dataset.size() > 1) {
            // Deterministic epoch-specific shuffling: reproducible across runs,
            // but avoids presenting AdamW with the same minibatch order forever.
            const auto epoch_seed = static_cast<std::mt19937::result_type>(
                static_cast<unsigned long long>(split_seed) +
                1000003ULL * static_cast<unsigned long long>(epoch));
            std::mt19937 epoch_rng(epoch_seed);
            std::shuffle(train_dataset.begin(), train_dataset.end(), epoch_rng);
        }
        model->train();
        float train_loss_tot = 0.0f;
        float train_loss_f_norm_tot = 0.0f;
        float train_loss_t_norm_tot = 0.0f;
        float train_mae_forces_tot = 0.0f;  
        float train_mae_torques_tot = 0.0f; 
        int train_torque_frames = 0; 

        std::vector<CGFrame> train_batch_frames;

        printf("Training:\n");
        for (size_t i = 0; i < train_dataset.size(); ++i) {
            train_batch_frames.push_back(train_dataset[i]);
            if (train_batch_frames.size() == batch_size || i == train_dataset.size() - 1) {
                
#ifdef __APPLE__
                void* pool = objc_autoreleasePoolPush();
#endif

                { // --- INIZIO SCOPE TENSORI ---
                    // Tutti i tensori locali verranno distrutti alla fine di questo blocco
                    // PRIMA di chiamare objc_autoreleasePoolPop()!

                    optimizer.zero_grad();
                
                    CGBatch batch = collate_batch(train_batch_frames, cutoff, device);

                    auto row = batch.edge_index[0];
                    auto col = batch.edge_index[1];

                    torch::Tensor pos_row = batch.coordinates.index_select(0, row);
                    torch::Tensor pos_col = batch.coordinates.index_select(0, col);
                    
                    auto edge_batch_indices = batch.batch_indices.index({row});
                    auto edge_boxes = batch.frame_boxes.index({edge_batch_indices});
                    
                    // Correzione PBC numerica (detach per non portare round nel grafo)
                    auto r_ij_val = pos_row - pos_col;
                    r_ij_val = r_ij_val - edge_boxes * torch::round(r_ij_val / edge_boxes).detach();

                    // --- PASSO UNICO: calcolo forze (con grafo autograd per i pesi) ---
                    torch::Tensor r_ij_leaf = r_ij_val.requires_grad_(true);
                    torch::Tensor pmf_diff = model->forward_with_rij(batch.site_types, r_ij_leaf, batch.edge_index, batch.batch_indices);
                    
                    auto g_diff = torch::autograd::grad({pmf_diff}, {r_ij_leaf},
                        {torch::ones_like(pmf_diff)}, /*create_graph=*/true, /*retain_graph=*/true);
                    torch::Tensor f_diff = -g_diff[0];  // [N_edges, 3] - con grafo

                    // Aggregazione forze molecolari
                    auto mol_of_row = batch.mol_indices.index_select(0, row);
                    auto mol_of_col = batch.mol_indices.index_select(0, col);
                    
                    torch::Tensor pred_mol_forces_diff = torch::zeros({batch.num_molecules_in_batch, 3}, f_diff.options());
                    pred_mol_forces_diff.index_add_(0, mol_of_row,  f_diff);
                    pred_mol_forces_diff.index_add_(0, mol_of_col, -f_diff);

                    torch::Tensor loss_f_raw = torch::mse_loss(
                        pred_mol_forces_diff,
                        batch.target_mol_forces
                    );
                    torch::Tensor loss_f_norm = loss_f_raw / force_scale2;

                    // Momenti torcenti (Ripristinati senza forward pass addizionali)
                    torch::Tensor site_centers = batch.mol_centers.index({batch.mol_indices});
                    torch::Tensor site_f_per_site = torch::zeros({(long)batch.coordinates.size(0), 3}, f_diff.options());
                    site_f_per_site.index_add_(0, row,  f_diff);
                    site_f_per_site.index_add_(0, col, -f_diff);
                    auto site_boxes = batch.frame_boxes.index({batch.batch_indices});
                    torch::Tensor r_vec = (batch.coordinates - site_centers).detach();
                    r_vec = r_vec - site_boxes * torch::round(r_vec / site_boxes).detach();
                    torch::Tensor site_torques = torch::linalg_cross(r_vec, site_f_per_site);
                    torch::Tensor pred_mol_torques = torch::zeros({batch.num_molecules_in_batch, 3}, site_torques.options());
                    pred_mol_torques.index_add_(0, batch.mol_indices, site_torques);

                    auto mol_indices_long = batch.mol_indices.to(torch::kLong);
                    torch::Tensor sites_per_mol = torch::bincount(mol_indices_long, torch::Tensor(), batch.num_molecules_in_batch);
                    torch::Tensor torque_mask = (sites_per_mol > 1).to(torch::kFloat32); 
                    float num_valid_mols = torque_mask.sum().item<float>();

                    torch::Tensor loss_t_norm = torch::zeros({}, loss_f_norm.options());
                    torch::Tensor loss_final = loss_f_norm;
                    if (num_valid_mols > 0) {
                        torch::Tensor loss_t_raw = torch::mse_loss(
                            pred_mol_torques, batch.target_mol_torques, torch::Reduction::None);
                        torch::Tensor loss_t_masked = loss_t_raw * torque_mask.unsqueeze(-1);
                        torch::Tensor loss_t_phys = loss_t_masked.sum() / (num_valid_mols * 3.0f);
                        loss_t_norm = loss_t_phys / torque_scale2;
                        loss_final = loss_final + torque_weight * loss_t_norm;
                    }
                    if (lipschitz_lambda > 0.0f) {
                        torch::Tensor loss_lipschitz =
                            site_f_per_site.norm(2, 1).pow(2).mean() / force_scale2;
                        loss_final = loss_final + lipschitz_lambda * loss_lipschitz;
                    }

                    loss_final.backward();

                    if (grad_clip_norm > 0.0f) {
                        torch::nn::utils::clip_grad_norm_(
                            model->parameters(), /*max_norm=*/ grad_clip_norm);
                    }
                    optimizer.step();

                    float current_batch_weight = static_cast<float>(train_batch_frames.size());
                    float mae_f_phys = torch::l1_loss(pred_mol_forces_diff, batch.target_mol_forces).item<float>();
                    train_loss_tot        += loss_final.item<float>() * current_batch_weight;
                    train_loss_f_norm_tot += loss_f_norm.item<float>() * current_batch_weight;
                    train_loss_t_norm_tot += loss_t_norm.item<float>() * current_batch_weight;
                    train_mae_forces_tot  += mae_f_phys                * current_batch_weight; 
                    
                    if (num_valid_mols > 0) {
                        torch::Tensor abs_t = torch::abs(
                            pred_mol_torques - batch.target_mol_torques
                        ) * torque_mask.unsqueeze(-1);
                        float masked_mae_t = abs_t.sum().item<float>() /
                                             (num_valid_mols * 3.0f);
                        train_mae_torques_tot += masked_mae_t * current_batch_weight;
                        train_torque_frames   += train_batch_frames.size();
                    }

#ifdef __APPLE__
                    if (torch::mps::is_available()) {
                        torch::mps::synchronize();
                        at::mps::getIMPSAllocator()->emptyCache();
                    }
#endif
                } // --- FINE SCOPE TENSORI ---

#ifdef __APPLE__
                objc_autoreleasePoolPop(pool);
#endif

                progress_bar(static_cast<double>(i + 1) / train_dataset.size());
                train_batch_frames.clear(); 
            }
        }

        // ---------------------------------------------------------
        // CICLO DI VALIDAZIONE
        // ---------------------------------------------------------
        model->eval(); 
        
        float val_loss_tot = 0.0f;
        float val_loss_f_norm_tot = 0.0f;
        float val_loss_t_norm_tot = 0.0f;
        float val_mae_forces_tot = 0.0f;
        float val_mae_torques_tot = 0.0f;
        int val_torque_frames = 0;

        std::vector<CGFrame> val_batch_frames;

        printf("Validation:\n");
        for (size_t i = 0; i < val_dataset.size(); ++i) {
            val_batch_frames.push_back(val_dataset[i]);

            if (val_batch_frames.size() == batch_size || i == val_dataset.size() - 1) {
                CGBatch batch = collate_batch(val_batch_frames, cutoff, device);

                auto row = batch.edge_index[0];
                auto col = batch.edge_index[1];

                torch::Tensor pos_row = batch.coordinates.index_select(0, row);
                torch::Tensor pos_col = batch.coordinates.index_select(0, col);
                
                auto edge_batch_indices = batch.batch_indices.index({row});
                auto edge_boxes = batch.frame_boxes.index({edge_batch_indices});
                
                auto r_ij_raw = pos_row - pos_col;
                auto r_ij = (r_ij_raw - edge_boxes * torch::round(r_ij_raw / edge_boxes).detach()).requires_grad_(true);

                torch::Tensor pred_pmf = model->forward_with_rij(batch.site_types, r_ij, batch.edge_index, batch.batch_indices);
                
                auto grad_outputs = torch::ones_like(pred_pmf);
                auto gradients = torch::autograd::grad({pred_pmf}, {r_ij}, {grad_outputs}, false, false);
                torch::Tensor f_r_ij = -gradients[0];

                torch::Tensor pred_mol_forces = torch::zeros({batch.num_molecules_in_batch, 3}, f_r_ij.options());
                auto mol_of_row = batch.mol_indices.index_select(0, row);
                auto mol_of_col = batch.mol_indices.index_select(0, col);
                pred_mol_forces.index_add_(0, mol_of_row,  f_r_ij);
                pred_mol_forces.index_add_(0, mol_of_col, -f_r_ij);

                torch::Tensor site_centers = batch.mol_centers.index({batch.mol_indices});
                torch::Tensor site_forces_per_site = torch::zeros_like(batch.coordinates);
                site_forces_per_site.index_add_(0, row,  f_r_ij);
                site_forces_per_site.index_add_(0, col, -f_r_ij);
                auto site_boxes = batch.frame_boxes.index({batch.batch_indices});
                torch::Tensor r_vec = batch.coordinates - site_centers;
                r_vec = r_vec - site_boxes * torch::round(r_vec / site_boxes).detach();
                torch::Tensor site_torques = torch::linalg_cross(r_vec, site_forces_per_site);
                torch::Tensor pred_mol_torques = torch::zeros({batch.num_molecules_in_batch, 3}, site_torques.options());
                pred_mol_torques.index_add_(0, batch.mol_indices, site_torques);

                auto mol_indices_long = batch.mol_indices.to(torch::kLong);
                torch::Tensor sites_per_mol = torch::bincount(mol_indices_long, torch::Tensor(), batch.num_molecules_in_batch);
                torch::Tensor torque_mask = (sites_per_mol > 1).to(torch::kFloat32);
                float num_valid_mols = torque_mask.sum().item<float>();

                torch::Tensor loss_f_raw = torch::mse_loss(
                    pred_mol_forces,
                    batch.target_mol_forces
                );
                torch::Tensor loss_f_norm = loss_f_raw / force_scale2;
                torch::Tensor loss_t_norm = torch::zeros({}, loss_f_norm.options());
                if (num_valid_mols > 0) {
                    torch::Tensor loss_t_raw = torch::mse_loss(
                        pred_mol_torques, batch.target_mol_torques, torch::Reduction::None);
                    torch::Tensor loss_t_masked = loss_t_raw * torque_mask.unsqueeze(-1);
                    torch::Tensor loss_t_phys = loss_t_masked.sum() / (num_valid_mols * 3.0f);
                    loss_t_norm = loss_t_phys / torque_scale2;
                }

                torch::Tensor loss = loss_f_norm + torque_weight * loss_t_norm;
                if (lipschitz_lambda > 0.0f) {
                    torch::Tensor loss_lipschitz =
                        site_forces_per_site.norm(2, 1).pow(2).mean() / force_scale2;
                    loss = loss + lipschitz_lambda * loss_lipschitz;
                }
                
                float current_batch_weight = static_cast<float>(val_batch_frames.size());
                float mae_f_phys = torch::l1_loss(pred_mol_forces, batch.target_mol_forces).item<float>();
                val_loss_tot        += loss.item<float>()        * current_batch_weight;
                val_loss_f_norm_tot += loss_f_norm.item<float>() * current_batch_weight;
                val_loss_t_norm_tot += loss_t_norm.item<float>() * current_batch_weight;
                val_mae_forces_tot  += mae_f_phys                 * current_batch_weight;
                
                if (num_valid_mols > 0) {
                    torch::Tensor abs_t = torch::abs(
                        pred_mol_torques - batch.target_mol_torques
                    ) * torque_mask.unsqueeze(-1);
                    float masked_mae_t = abs_t.sum().item<float>() /
                                         (num_valid_mols * 3.0f);
                    val_mae_torques_tot += masked_mae_t * current_batch_weight;
                    val_torque_frames   += val_batch_frames.size();
                }

                progress_bar(static_cast<double>(i + 1) / val_dataset.size());
                val_batch_frames.clear();
            }
        }
        std::cout << "\n";

        float train_loss_avg = train_loss_tot / train_dataset.size();
        float train_loss_f_norm_avg = train_loss_f_norm_tot / train_dataset.size();
        float train_loss_t_norm_avg = train_loss_t_norm_tot / train_dataset.size();
        float train_mae_forces_avg = train_mae_forces_tot / train_dataset.size();
        float train_mae_torques_avg = (train_torque_frames > 0) ? (train_mae_torques_tot / train_torque_frames) : 0.0f;

        float val_loss_avg = val_loss_tot / val_dataset.size();
        float val_loss_f_norm_avg = val_loss_f_norm_tot / val_dataset.size();
        float val_loss_t_norm_avg = val_loss_t_norm_tot / val_dataset.size();
        float val_mae_forces_avg = val_mae_forces_tot / val_dataset.size();
        float val_mae_torques_avg = (val_torque_frames > 0) ? (val_mae_torques_tot / val_torque_frames) : 0.0f;

        std::cout << "\nEpoca [" << epoch << "/" << max_epochs << "]\n"
                  << "  [LR]    " << current_lr << "\n"
                  << "  [TRAIN] Loss: " << train_loss_avg
                  << " (F: " << train_loss_f_norm_avg
                  << ", T: " << train_loss_t_norm_avg << ")"
                  << " | MAE Forze: " << train_mae_forces_avg
                  << " | MAE Torques: " << train_mae_torques_avg << "\n"
                  << "  [VAL]   Loss: " << val_loss_avg
                  << " (F: " << val_loss_f_norm_avg
                  << ", T: " << val_loss_t_norm_avg << ")"
                  << " | MAE Forze: " << val_mae_forces_avg
                  << " | MAE Torques: " << val_mae_torques_avg << "\n";

        if (csv_file.is_open()) {
            csv_file << epoch << ","
                     << train_loss_avg << "," << val_loss_avg << ","
                     << train_loss_f_norm_avg << "," << train_loss_t_norm_avg << ","
                     << val_loss_f_norm_avg << "," << val_loss_t_norm_avg << ","
                     << train_mae_forces_avg << "," << train_mae_torques_avg << ","
                     << val_mae_forces_avg << "," << val_mae_torques_avg << ","
                     << val_zero_f_norm << "," << val_zero_t_norm << ","
                     << val_zero_total << "\n";
            csv_file.flush();
        }

        if (val_loss_avg < best_val_loss) {
            best_val_loss = val_loss_avg;
            lr_counter = 0; 
        } else {
            lr_counter++;
            if (lr_counter >= lr_patience) {
                current_lr *= 0.5f; 
                if (current_lr < 1e-6f) {
                    current_lr = 1e-6f;
                }
                
                for (auto& param_group : optimizer.param_groups()) {
                    static_cast<torch::optim::AdamWOptions&>(param_group.options()).lr(current_lr);
                }
                std::cout << "  ---> [Scheduler] Plateau raggiunto. Learning Rate abbassato a: " << current_lr << "\n";
                lr_counter = 0; 
            }
        }
        
        early_stopping.check(model, val_loss_avg, device);
        if (early_stopping.early_stop) {
            std::cout << "[INFO] Addestramento interrotto (Early Stopping).\n";
            break;
        }
    }

    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Training ended without producing model file: " + model_path);
    }
    write_model_manifest(
        model_path, dataset_path, config_path, effective_config, early_stopping.best_loss);

    return 0;
}
