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
std::vector<CGFrame> read_cg_dataset(const std::string& filepath) {
    std::vector<CGFrame> dataset;
    std::ifstream file(filepath, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << filepath << "\n";
        return dataset;
    }

    int num_frames;
    file.read(reinterpret_cast<char*>(&num_frames), sizeof(int));
    dataset.reserve(num_frames);

    for (int f = 0; f < num_frames; ++f) {
        CGFrame frame;
        int num_molecules, num_total_sites;
        
        file.read(reinterpret_cast<char*>(&num_molecules), sizeof(int));
        file.read(reinterpret_cast<char*>(&num_total_sites), sizeof(int));
        file.read(reinterpret_cast<char*>(frame.box), 3 * sizeof(float));
        
        frame.molecules.reserve(num_molecules);

        for (int m = 0; m < num_molecules; ++m) {
            CGMolecule mol;
            int num_sites;
            
            file.read(reinterpret_cast<char*>(&mol.molecule_id), sizeof(int));
            file.read(reinterpret_cast<char*>(&num_sites), sizeof(int));
            
            file.read(reinterpret_cast<char*>(mol.center_of_geometry), 3 * sizeof(float));
            file.read(reinterpret_cast<char*>(mol.target_force), 3 * sizeof(float));
            file.read(reinterpret_cast<char*>(mol.target_torque), 3 * sizeof(float));
            
            mol.sites.reserve(num_sites);
            for (int s = 0; s < num_sites; ++s) {
                CGSite site;
                site.molecule_id = mol.molecule_id;
                file.read(reinterpret_cast<char*>(&site.site_type), sizeof(int));
                file.read(reinterpret_cast<char*>(&site.x), sizeof(float));
                file.read(reinterpret_cast<char*>(&site.y), sizeof(float));
                file.read(reinterpret_cast<char*>(&site.z), sizeof(float));
                
                mol.sites.push_back(site);
            }
            frame.molecules.push_back(mol);
        }
        dataset.push_back(frame);
    }
    
    file.close();
    std::cout << "[INFO] Letti " << dataset.size() << " frame dal dataset.\n";
    return dataset;
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
    if (manifest.value("schema_version", -1) != 1 ||
        manifest.value("framework", std::string()) != "MLCG_Framework_v2") {
        throw std::runtime_error("Unsupported model manifest: " + manifest_path);
    }
    const auto& architecture = manifest.at("architecture");
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
        {"num_species", effective_config.at("num_species")},
        {"hidden_channels", effective_config.at("hidden_channels")},
        {"n_layers", effective_config.at("n_layers")},
        {"num_rbf", effective_config.at("num_rbf")},
        {"cutoff", effective_config.at("cutoff")},
        {"toxvaerd_alpha", effective_config.at("toxvaerd_alpha")},
    };
    json manifest = {
        {"schema_version", 1},
        {"framework", "MLCG_Framework_v2"},
        {"architecture", architecture},
        {"effective_config", effective_config},
        {"model_path", model_path},
        {"model_file_size_bytes", file_size_or_zero(model_path)},
        {"dataset_path", dataset_path},
        {"dataset_file_size_bytes", file_size_or_zero(dataset_path)},
        {"config_path", config_path},
        {"config_file_size_bytes", file_size_or_zero(config_path)},
        {"split_seed", 42},
        {"validation_fraction", 0.2},
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
        if (loaded_config.contains("batch_size")) batch_size = loaded_config["batch_size"];
        if (batch_size <= 0) {
            throw std::runtime_error("batch_size must be positive");
        }
        std::cout << "[INFO] Caricati iperparametri da " << config_path << "\n";
    } else {
        std::cerr << "[ERROR] Cannot read required training config: " << config_path << "\n";
        return 2;
    }

    json effective_config = loaded_config;
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
    effective_config["batch_size"] = batch_size;

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
        csv_file << "Epoch,Train_Loss,Val_Loss,Train_MAE_F,Train_MAE_T,Val_MAE_F,Val_MAE_T\n";
    }

    std::cout << "\n[INFO] Caricamento dataset binario in corso: " << dataset_path << "...\n";
    
    std::vector<CGFrame> full_dataset = read_cg_dataset(dataset_path);
    
    if (full_dataset.empty()) {
        std::cerr << "Errore critico: dataset vuoto o file non trovato. Interruzione.\n";
        return 1;
    }

    std::mt19937 g(42);
    std::shuffle(full_dataset.begin(), full_dataset.end(), g);

    // Divisione 80% Train, 20% Validation
    size_t total_frames = full_dataset.size();
    size_t val_size = total_frames * 0.2;
    size_t train_size = total_frames - val_size;

    std::vector<CGFrame> train_dataset(full_dataset.begin(), full_dataset.begin() + train_size);
    std::vector<CGFrame> val_dataset(full_dataset.begin() + train_size, full_dataset.end());

    std::cout << "[INFO] Split completato:\n"
              << "       - Train: " << train_dataset.size() << " frames\n"
              << "       - Val:   " << val_dataset.size() << " frames\n\n";

    // -------------------------------------------------------
    // Calcolo della standardizzazione delle forze sul train set
    // -------------------------------------------------------
    double force_sum2 = 0.0;
    long   force_count = 0;
    for (const auto& frame : train_dataset)
        for (const auto& mol : frame.molecules)
            for (int k = 0; k < 3; ++k) {
                force_sum2 += mol.target_force[k] * mol.target_force[k];
                force_count++;
            }
    float force_std = (force_count > 0) ? std::sqrt(force_sum2 / force_count) : 1.0f;
    if (force_std < 1e-6f) force_std = 1.0f;
    std::cout << "[INFO] Force std (train): " << force_std << " kJ/(mol*nm)\n\n";

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        model->train();
        float train_loss_tot = 0.0f;
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

                    torch::Tensor loss_f_diff = torch::mse_loss(
                        pred_mol_forces_diff,
                        batch.target_mol_forces
                    );

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

                    torch::Tensor loss_final = loss_f_diff;
                    if (num_valid_mols > 0) {
                        torch::Tensor loss_t_raw = torch::mse_loss(pred_mol_torques, batch.target_mol_torques, torch::Reduction::None);
                        torch::Tensor loss_t_masked = loss_t_raw * torque_mask.unsqueeze(-1);
                        torch::Tensor loss_t = loss_t_masked.sum() / (num_valid_mols * 3.0f);
                        loss_final = loss_final + torque_weight * loss_t;
                    }
                    if (lipschitz_lambda > 0.0f) {
                        torch::Tensor loss_lipschitz = site_f_per_site.norm(2, 1).pow(2).mean();
                        loss_final = loss_final + lipschitz_lambda * loss_lipschitz;
                    }

                    loss_final.backward();

                    torch::nn::utils::clip_grad_norm_(model->parameters(), /*max_norm=*/ 1.0);
                    optimizer.step();

                    float current_batch_weight = static_cast<float>(train_batch_frames.size());
                    float mae_f_phys = torch::l1_loss(pred_mol_forces_diff, batch.target_mol_forces).item<float>();
                    train_loss_tot        += loss_final.item<float>() * current_batch_weight;
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

                torch::Tensor loss_f = torch::mse_loss(
                    pred_mol_forces,
                    batch.target_mol_forces
                );
                torch::Tensor loss_t;
                if (num_valid_mols > 0) {
                    torch::Tensor loss_t_raw = torch::mse_loss(pred_mol_torques, batch.target_mol_torques, torch::Reduction::None);
                    torch::Tensor loss_t_masked = loss_t_raw * torque_mask.unsqueeze(-1);
                    loss_t = loss_t_masked.sum() / (num_valid_mols * 3.0f);
                } else {
                    loss_t = torch::zeros({}, loss_f.options());
                }

                torch::Tensor loss = loss_f + torque_weight * loss_t;
                if (lipschitz_lambda > 0.0f) {
                    torch::Tensor loss_lipschitz = site_forces_per_site.norm(2, 1).pow(2).mean();
                    loss = loss + lipschitz_lambda * loss_lipschitz;
                }
                
                float current_batch_weight = static_cast<float>(val_batch_frames.size());
                float mae_f_phys = torch::l1_loss(pred_mol_forces, batch.target_mol_forces).item<float>();
                val_loss_tot       += loss.item<float>()  * current_batch_weight;
                val_mae_forces_tot += mae_f_phys          * current_batch_weight;
                
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
        float train_mae_forces_avg = train_mae_forces_tot / train_dataset.size();
        float train_mae_torques_avg = (train_torque_frames > 0) ? (train_mae_torques_tot / train_torque_frames) : 0.0f;   
   
        float val_loss_avg = val_loss_tot / val_dataset.size(); 
        float val_mae_forces_avg = val_mae_forces_tot / val_dataset.size();
        float val_mae_torques_avg = (val_torque_frames > 0) ? (val_mae_torques_tot / val_torque_frames) : 0.0f;

        std::cout << "\nEpoca [" << epoch << "/" << max_epochs << "]\n"
                  << "  [LR]    " << current_lr << "\n"
                  << "  [TRAIN] Loss: " << train_loss_avg 
                  << " | MAE Forze: " << train_mae_forces_avg 
                  << " | MAE Torques: " << train_mae_torques_avg << "\n"
                  << "  [VAL]   Loss: " << val_loss_avg 
                  << " | MAE Forze: " << val_mae_forces_avg 
                  << " | MAE Torques: " << val_mae_torques_avg << "\n";

        if (csv_file.is_open()) {
            csv_file << epoch << "," 
                     << train_loss_avg << "," << val_loss_avg << "," 
                     << train_mae_forces_avg << "," << train_mae_torques_avg << "," 
                     << val_mae_forces_avg << "," << val_mae_torques_avg << "\n";
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
