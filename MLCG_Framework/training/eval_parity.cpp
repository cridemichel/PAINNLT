#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <cstdint>
#include <limits>
#include <random>    // Aggiungi questo in cima al file per std::shuffle
#include <algorithm> // Aggiungi questo in cima per std::shuffle

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
using MoleculePairSet = std::unordered_set<std::uint64_t>;

std::uint64_t molecule_pair_key(int a, int b) {
    auto lo = static_cast<std::uint32_t>(std::min(a, b));
    auto hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | hi;
}

std::string resolve_config_relative_path(const std::string& config_path, const std::string& value) {
    if (value.empty()) return value;
    std::filesystem::path p(value);
    if (p.is_absolute()) return p.string();
    return (std::filesystem::path(config_path).parent_path() / p).lexically_normal().string();
}

MoleculePairSet load_excluded_molecule_pairs(const std::string& priors_path) {
    MoleculePairSet result;
    if (priors_path.empty()) return result;

    std::ifstream input(priors_path);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open exclusion priors: " + priors_path);
    }
    json priors;
    input >> priors;

    for (const auto& bond : priors.value("bonds", json::array())) {
        int a = bond.at("mol_i").get<int>();
        int b = bond.at("mol_j").get<int>();
        if (a != b) result.insert(molecule_pair_key(a, b));
    }
    for (const auto& angle : priors.value("angles", json::array())) {
        int a = angle.at("mol_i").get<int>();
        int b = angle.at("mol_k").get<int>();
        if (a != b) result.insert(molecule_pair_key(a, b));
    }
    return result;
}

CGBatch collate_batch(const std::vector<CGFrame>& frames, float cutoff, torch::Device device,
                      const MoleculePairSet& excluded_molecule_pairs) {
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
                // Match the ESPResSo graph exactly: exclude intra-molecular,
                // bonded 1-2, and angular 1-3 molecule pairs.
                const int mol_i = frame_sites[i].molecule_id;
                const int mol_j = frame_sites[j].molecule_id;
                if (mol_i == mol_j) continue;
                if (excluded_molecule_pairs.count(molecule_pair_key(mol_i, mol_j)) != 0) continue;
                
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
 
#ifdef __APPLE__
extern "C" {
    void* objc_autoreleasePoolPush(void);
    void objc_autoreleasePoolPop(void* pool);
}
#endif

int main(int argc, char** argv) {
    std::string dataset_file = "cg_dataset.bin";
    std::string model_file = "cg_model.pt";
    std::string config_file = "cg_model_config.json";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) dataset_file = argv[++i];
        if (arg == "--model" && i + 1 < argc) model_file = argv[++i];
        if (arg == "--config" && i + 1 < argc) config_file = argv[++i];
    }
    
    std::ifstream cfg_in(config_file);
    if (!cfg_in.is_open()) {
        std::cerr << "Config non trovato: " << config_file << "\n";
        return 1;
    }
    json config;
    cfg_in >> config;
    float cutoff = config.value("cutoff", 1.2f);
    const std::string exclusion_priors_path = resolve_config_relative_path(
        config_file, config.value("exclusion_priors", std::string())
    );
    MoleculePairSet excluded_molecule_pairs;
    try {
        excluded_molecule_pairs = load_excluded_molecule_pairs(exclusion_priors_path);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
    std::cout << "Excluded 1-2/1-3 molecule pairs: "
              << excluded_molecule_pairs.size() << "\n";
    
    std::vector<CGFrame> dataset = read_cg_dataset(dataset_file);
    if (dataset.empty()) return 1;
    
    // Reproduce the exact deterministic split used by train_painn.cpp.
    std::mt19937 split_rng(42);
    std::shuffle(dataset.begin(), dataset.end(), split_rng);
    size_t val_size = dataset.size() * 0.2;
    size_t train_size = dataset.size() - val_size;
    std::vector<CGFrame> val_dataset(dataset.begin() + train_size, dataset.end());
    
    torch::Device device = torch::kCPU;
#ifdef __APPLE__
    device = torch::Device(torch::kMPS);
#elif defined(__linux__)
    if (torch::cuda::is_available()) device = torch::Device(torch::kCUDA);
#endif
    std::cout << "Evaluating on " << (device.type() == torch::kMPS ? "MPS" : (device.type() == torch::kCUDA ? "CUDA" : "CPU")) << "...\n";
    
    PaiNNModel model(
        config["num_species"], config["hidden_channels"], config["n_layers"], 
        config["num_rbf"], cutoff, config.value("apply_envelope", true), 
        config.value("use_bias", false), config.value("toxvaerd_alpha", 0.1)
    );
    
    torch::load(model, model_file);
    model->to(device);
    model->eval();
    
    std::ofstream out_csv("parity_forces.csv");
    out_csv << "F_target_x,F_target_y,F_target_z,F_pred_x,F_pred_y,F_pred_z\n";
    
    size_t batch_size = config.value("batch_size", static_cast<size_t>(5));
    std::vector<CGFrame> val_batch_frames;
    
    int progress = 0;
    for (size_t i = 0; i < val_dataset.size(); ++i) {
        val_batch_frames.push_back(val_dataset[i]);
        if (val_batch_frames.size() == batch_size || i == val_dataset.size() - 1) {
            CGBatch batch = collate_batch(val_batch_frames, cutoff, device, excluded_molecule_pairs);
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
            
            auto pred_f_cpu = pred_mol_forces.cpu();
            auto target_f_cpu = batch.target_mol_forces.cpu();
            auto pred_f_acc = pred_f_cpu.accessor<float, 2>();
            auto target_f_acc = target_f_cpu.accessor<float, 2>();
            
            for(int m = 0; m < batch.num_molecules_in_batch; ++m) {
                out_csv << target_f_acc[m][0] << "," << target_f_acc[m][1] << "," << target_f_acc[m][2] << ","
                        << pred_f_acc[m][0] << "," << pred_f_acc[m][1] << "," << pred_f_acc[m][2] << "\n";
            }
            val_batch_frames.clear();
            
            progress++;
            if(progress % 10 == 0) std::cout << "Evaluated " << i << "/" << val_dataset.size() << " frames...\n";
        }
    }
    out_csv.close();
    std::cout << "Done! parity_forces.csv written.\n";
    return 0;
}
