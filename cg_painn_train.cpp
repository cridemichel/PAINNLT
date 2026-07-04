#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <limits>
#include <random>    // Aggiungi questo in cima al file per std::shuffle
#include <algorithm> // Aggiungi questo in cima per std::shuffle


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
    
    int site_offset = 0;
    int global_mol_idx = 0;
    float cutoff_sq = cutoff * cutoff;

    for (size_t b_idx = 0; b_idx < frames.size(); ++b_idx) {
        const auto& frame = frames[b_idx];
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
        
        // COSTRUZIONE DEL GRAFO (Escludendo gli archi intra-molecolari)
        int num_sites_in_frame = frame_sites.size();
        for (int i = 0; i < num_sites_in_frame; ++i) {
            for (int j = i + 1; j < num_sites_in_frame; ++j) {
                // REGOLA: Se appartengono alla stessa molecola, NON interagiscono
                if (frame_sites[i].molecule_id == frame_sites[j].molecule_id) continue;
                
                float dx = frame_sites[i].x - frame_sites[j].x;
                float dy = frame_sites[i].y - frame_sites[j].y;
                float dz = frame_sites[i].z - frame_sites[j].z;
                if ((dx*dx + dy*dy + dz*dz) <= cutoff_sq) {
                    edge_rows.push_back(frame_site_start + i); edge_cols.push_back(frame_site_start + j);
                    edge_rows.push_back(frame_site_start + j); edge_cols.push_back(frame_site_start + i);
                }
            }
        }
        site_offset += num_sites_in_frame;
    }
    
    // Creazione Tensori PyTorch
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

    void check(PaiNNModel& model, float val_loss) { 
        if (val_loss < best_loss) {
            best_loss = val_loss;
            counter = 0;
            torch::save(model, save_path);
            std::cout << "   ---> [Early Stopping] Miglioramento! Modello salvato.\n";
        } else {
            counter++;
            std::cout << "   ---> [Early Stopping] Nessun miglioramento (" << counter << "/" << patience << ").\n";
            if (counter >= patience) early_stop = true;
        }
    }
};

// =====================================================================
// 4. MAIN PROGRAM
// =====================================================================
int main() {
    torch::manual_seed(42);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Utilizzando il device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    // 1. Parametri Rete
    int num_atom_types = 100; // Capienza dizionario siti
    int dim = 128;
    int layers = 3;
    int num_rbf = 40; 
    float cutoff = 5.0f;
    std::string model_path = "best_cg_model.pt";

    // Salvataggio JSON (Una volta sola all'inizio)
    std::string json_path = model_path.substr(0, model_path.find_last_of('.')) + "_config.json";
    std::ofstream json_file(json_path);
    if (json_file.is_open()) {
        json_file << "{\n  \"num_atoms\": " << num_atom_types << ",\n  \"hidden_channels\": " << dim 
                  << ",\n  \"n_layers\": " << layers << ",\n  \"num_rbf\": " << num_rbf 
                  << ",\n  \"cutoff\": " << cutoff << "\n}\n";
        json_file.close();
        std::cout << "[INFO] File configurazione JSON salvato.\n";
    }

    // Inizializza il Modello
    PaiNNModel model(num_atom_types, dim, layers, num_rbf, cutoff);
    model->to(device);

    // Iperparametri Training
    float initial_lr = 5e-4;
    float torque_weight = 1.0f; // Peso relativo del torque rispetto alla forza nella loss
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(initial_lr).weight_decay(1e-5));
    EarlyStopping early_stopping(15, model_path);

    std::ofstream csv_file("training_log.csv");
    if (csv_file.is_open()) {
        csv_file << "Epoch,Train_Loss,Val_Loss,Val_MAE_Forces,Val_MAE_Torques\n";
    }

    // =====================================================================
    // LETTURA E PREPARAZIONE DEL DATASET
    // =====================================================================
    std::cout << "\n[INFO] Caricamento dataset binario in corso...\n";
    std::string dataset_path = "cg_dataset.bin"; // Il file generato da Python
    
    std::vector<CGFrame> full_dataset = read_cg_dataset(dataset_path);
    
    if (full_dataset.empty()) {
        std::cerr << "Errore critico: dataset vuoto o file non trovato. Interruzione.\n";
        return 1;
    }

    // Mescoliamo i frame per evitare bias sequenziali (Fondamentale nel ML!)
    // GROMACS genera traiettorie correlate nel tempo, mescolarle rompe questa correlazione.
    std::random_device rd;
    std::mt19937 g(rd());
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

    int max_epochs = 500;
    
    // ---------------------------------------------------------
    // CICLO DI ADDESTRAMENTO
    // ---------------------------------------------------------
    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        model->train();
        float train_loss_tot = 0.0f;

        // Esempio logica batch (Assumiamo 1 frame per iterazione per semplicità nel template)
        for (const auto& frame : train_dataset) {
            optimizer.zero_grad();
            
            std::vector<CGFrame> batch_frames = {frame};
            CGBatch batch = collate_batch(batch_frames, cutoff, device);
            batch.coordinates.set_requires_grad(true);

            // Calcolo esplicito di r_ij dalle coordinate (necessario per l'autograd)
            auto row = batch.edge_index[0];
            auto col = batch.edge_index[1];
            auto r_ij = batch.coordinates.index({row}) - batch.coordinates.index({col});

            // Forward della rete: predice un'energia fittizia del PMF
            torch::Tensor pred_pmf = model->forward_with_rij(batch.site_types, r_ij, batch.edge_index, batch.batch_indices);
            
            // 1. Calcolo Forze sui singoli SITI (F = - grad(E))
            auto grad_outputs = torch::ones_like(pred_pmf);
            auto gradients = torch::autograd::grad({pred_pmf}, {batch.coordinates}, {grad_outputs}, true, true);
            torch::Tensor site_forces = -gradients[0]; // [N_sites, 3]

            // 2. Aggregazione Forze Molecolari (Vettorializzata)
            torch::Tensor pred_mol_forces = torch::zeros({batch.num_molecules_in_batch, 3}, site_forces.options());
            pred_mol_forces.index_add_(0, batch.mol_indices, site_forces);

            // 3. Calcolo e Aggregazione Momenti Torcenti
            // Mappiamo i centri molecolari su ogni sito
            torch::Tensor site_centers = batch.mol_centers.index({batch.mol_indices});
            torch::Tensor r_vec = batch.coordinates - site_centers; // Vettore braccio
            torch::Tensor site_torques = torch::linalg_cross(r_vec, site_forces);
            
            torch::Tensor pred_mol_torques = torch::zeros({batch.num_molecules_in_batch, 3}, site_torques.options());
            pred_mol_torques.index_add_(0, batch.mol_indices, site_torques);

            // 4. LOSS (Forze + Momenti Torcenti, NESSUNA ENERGIA)
            torch::Tensor loss_f = torch::mse_loss(pred_mol_forces, batch.target_mol_forces);
            torch::Tensor loss_t = torch::mse_loss(pred_mol_torques, batch.target_mol_torques);
            torch::Tensor loss = loss_f + torque_weight * loss_t;

            loss.backward();
            optimizer.step();
            train_loss_tot += loss.item<float>();
        }

        // ---------------------------------------------------------
        // CICLO DI VALIDAZIONE
        // ---------------------------------------------------------
        model->eval(); // Disabilita Dropout/BatchNorm (se presenti)
        
        float val_loss_tot = 0.0f;
        float val_mae_forces_tot = 0.0f;
        float val_mae_torques_tot = 0.0f;

        // Iteriamo sul dataset di validazione
        for (const auto& frame : val_dataset) {
            std::vector<CGFrame> batch_frames = {frame};
            CGBatch batch = collate_batch(batch_frames, cutoff, device);
            
            // FONDAMENTALE: Richiediamo il gradiente sulle coordinate anche in validazione
            // altrimenti torch::autograd::grad fallirà!
            batch.coordinates.set_requires_grad(true);

            auto row = batch.edge_index[0];
            auto col = batch.edge_index[1];
            auto r_ij = batch.coordinates.index({row}) - batch.coordinates.index({col});

            torch::Tensor pred_pmf = model->forward_with_rij(batch.site_types, r_ij, batch.edge_index, batch.batch_indices);
            
            // 1. Calcolo Forze
            auto grad_outputs = torch::ones_like(pred_pmf);
            auto gradients = torch::autograd::grad({pred_pmf}, {batch.coordinates}, {grad_outputs}, true, true);
            torch::Tensor site_forces = -gradients[0];

            // 2. Aggregazione Forze Molecolari
            torch::Tensor pred_mol_forces = torch::zeros({batch.num_molecules_in_batch, 3}, site_forces.options());
            pred_mol_forces.index_add_(0, batch.mol_indices, site_forces);

            // 3. Calcolo e Aggregazione Momenti Torcenti
            torch::Tensor site_centers = batch.mol_centers.index({batch.mol_indices});
            torch::Tensor r_vec = batch.coordinates - site_centers; 
            torch::Tensor site_torques = torch::linalg_cross(r_vec, site_forces);
            
            torch::Tensor pred_mol_torques = torch::zeros({batch.num_molecules_in_batch, 3}, site_torques.options());
            pred_mol_torques.index_add_(0, batch.mol_indices, site_torques);

            // 4. Calcolo delle Metriche (Senza backward!)
            torch::Tensor loss_f = torch::mse_loss(pred_mol_forces, batch.target_mol_forces);
            torch::Tensor loss_t = torch::mse_loss(pred_mol_torques, batch.target_mol_torques);
            torch::Tensor loss = loss_f + torque_weight * loss_t;

            val_loss_tot += loss.item<float>();
            
            // Calcolo del Mean Absolute Error (MAE) per i log
            val_mae_forces_tot += torch::l1_loss(pred_mol_forces, batch.target_mol_forces).item<float>();
            val_mae_torques_tot += torch::l1_loss(pred_mol_torques, batch.target_mol_torques).item<float>();
        }

        // 1. Calcolo delle medie per l'epoca
        // Dividiamo la somma totale per il numero effettivo di frame in ciascun dataset
        float train_loss_avg = train_loss_tot / train_dataset.size(); 
        
        float val_loss_avg = val_loss_tot / val_dataset.size(); 
        float val_mae_forces_avg = val_mae_forces_tot / val_dataset.size();
        float val_mae_torques_avg = val_mae_torques_tot / val_dataset.size();

        // 2. LOGGING SU SCHERMO (Usando solo le medie!)
        std::cout << "Epoca [" << epoch << "/" << max_epochs << "]\n"
                  << "  [TRAIN] Loss Media: " << train_loss_avg << "\n"
                  << "  [VAL]   Loss Media: " << val_loss_avg 
                  << " | MAE Forze (Mol): " << val_mae_forces_avg 
                  << " | MAE Torques: " << val_mae_torques_avg << "\n";

        // 3. SALVATAGGIO NEL CSV (Usando solo le medie!)
        if (csv_file.is_open()) {
            csv_file << epoch << "," 
                     << train_loss_avg << "," 
                     << val_loss_avg << "," 
                     << val_mae_forces_avg << "," 
                     << val_mae_torques_avg << "\n";
            csv_file.flush();
        }
        early_stopping.check(model, val_loss_avg);
        if (early_stopping.early_stop) {
            std::cout << "[INFO] Addestramento interrotto (Early Stopping).\n";
            break;
        }
    }

    return 0;
}
