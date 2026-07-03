#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <limits> 
#include <cmath>

// =====================================================================
// 1. STRUTTURE DATI E BATCHING
// =====================================================================
struct MoleculeFrame {
    torch::Tensor atomic_numbers;
    torch::Tensor coordinates;
    torch::Tensor energy;
    torch::Tensor forces;
};

struct PaiNNBatch {
    torch::Tensor atomic_numbers;
    torch::Tensor coordinates;
    torch::Tensor edge_index;
    torch::Tensor batch_indices;
    torch::Tensor energy_true;
    torch::Tensor forces_true;
};

PaiNNBatch collate_batch(const std::vector<MoleculeFrame>& frames, float cutoff, torch::Device device) {
    std::vector<torch::Tensor> list_atoms, list_coords, list_edges, list_batch, list_energies, list_forces;
    int atom_offset = 0;

    for (size_t b_idx = 0; b_idx < frames.size(); ++b_idx) {
        auto& frame = frames[b_idx];
        int num_atoms = frame.atomic_numbers.size(0);

        list_atoms.push_back(frame.atomic_numbers);
        list_coords.push_back(frame.coordinates);
        list_energies.push_back(frame.energy);
        list_forces.push_back(frame.forces);
        list_batch.push_back(torch::full({num_atoms}, (int64_t)b_idx, torch::kInt64));

        auto coords = frame.coordinates;
        auto r_diff = coords.unsqueeze(1) - coords.unsqueeze(0); 
        auto dists = torch::norm(r_diff, 2, /*dim=*/2);

        auto mask = (dists < cutoff) & (torch::eye(num_atoms, dists.options()).logical_not());
        auto edges = torch::nonzero(mask).t();

        list_edges.push_back(edges + atom_offset);
        atom_offset += num_atoms;
    }

    PaiNNBatch batch;
    batch.atomic_numbers = torch::cat(list_atoms, 0).to(device);
    batch.coordinates    = torch::cat(list_coords, 0).to(device);
    batch.edge_index     = torch::cat(list_edges, 1).to(device);
    batch.batch_indices  = torch::cat(list_batch, 0).to(device);
    batch.energy_true    = torch::cat(list_energies, 0).reshape({(long)frames.size(), 1}).to(device);
    batch.forces_true    = torch::cat(list_forces, 0).to(device);

    return batch;
}

std::vector<MoleculeFrame> load_md17_binary(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Impossibile aprire: " + filepath);

    int num_frames = 0, num_atoms = 0;
    file.read(reinterpret_cast<char*>(&num_frames), sizeof(int));
    file.read(reinterpret_cast<char*>(&num_atoms), sizeof(int));

    std::cout << "Lettura di " << filepath << ": " << num_frames << " frame, " << num_atoms << " atomi.\n";

    std::vector<int> atomic_numbers_raw(num_atoms);
    std::vector<MoleculeFrame> dataset;
    dataset.reserve(num_frames);

    for (int f = 0; f < num_frames; ++f) {
        double energy_raw; 
        file.read(reinterpret_cast<char*>(&energy_raw), sizeof(double));
        
        file.read(reinterpret_cast<char*>(atomic_numbers_raw.data()), num_atoms * sizeof(int));

        std::vector<double> coords_raw(num_atoms * 3);
        file.read(reinterpret_cast<char*>(coords_raw.data()), num_atoms * 3 * sizeof(double));

        std::vector<double> forces_raw(num_atoms * 3);
        file.read(reinterpret_cast<char*>(forces_raw.data()), num_atoms * 3 * sizeof(double));

        MoleculeFrame frame;
        std::vector<int64_t> atom_types_64(atomic_numbers_raw.begin(), atomic_numbers_raw.end());
        frame.atomic_numbers = torch::tensor(atom_types_64, torch::kInt64);
        
        frame.coordinates    = torch::tensor(coords_raw, torch::kFloat64).reshape({num_atoms, 3}).to(torch::kFloat32);
        frame.energy         = torch::tensor({energy_raw}, torch::kFloat64).to(torch::kFloat32);
        frame.forces         = torch::tensor(forces_raw, torch::kFloat64).reshape({num_atoms, 3}).to(torch::kFloat32);
        
        dataset.push_back(frame);
    }
    return dataset;
}

// =====================================================================
// 2. ARCHITETTURA PAINN (Message, Update, Model)
// =====================================================================
struct PaiNNMessageImpl : torch::nn::Module {
    torch::nn::Linear scalar_mlp{nullptr}, filter_mlp{nullptr};
    PaiNNMessageImpl(int dim) {
        scalar_mlp = register_module("scalar_mlp", torch::nn::Linear(dim, dim * 3));
        filter_mlp = register_module("filter_mlp", torch::nn::Linear(20, dim * 3));
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor s, torch::Tensor v, torch::Tensor edge_index, torch::Tensor rbf, torch::Tensor r_ij_norm) {
        auto row = edge_index[0], col = edge_index[1]; 
        auto w = filter_mlp->forward(rbf); 
        auto interaction = scalar_mlp->forward(s.index({row})) * w;      
        auto chunks = interaction.chunk(3, 1);
        auto delta_v_edges = v.index({row}) * chunks[1].unsqueeze(1) + chunks[2].unsqueeze(1) * r_ij_norm.unsqueeze(2);
        auto delta_s = torch::zeros_like(s);
        auto delta_v = torch::zeros_like(v);
        delta_s.index_add_(0, col, chunks[0]);
        delta_v.index_add_(0, col, delta_v_edges);
        return {delta_s, delta_v};
    }
};
TORCH_MODULE(PaiNNMessage);

struct PaiNNUpdateImpl : torch::nn::Module {
    torch::nn::Linear linear_v{nullptr}, linear_u{nullptr};
    torch::nn::Sequential scalar_mlp{nullptr};
    PaiNNUpdateImpl(int dim) {
        linear_v = register_module("linear_v", torch::nn::Linear(dim, dim));
        linear_u = register_module("linear_u", torch::nn::Linear(dim, dim));
        scalar_mlp = register_module("scalar_mlp", torch::nn::Sequential(
            torch::nn::Linear(dim * 2, dim), torch::nn::SiLU(), torch::nn::Linear(dim, dim * 3)
        ));
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor s, torch::Tensor v) {
        auto v_v = linear_v->forward(v); 
        auto v_u = linear_u->forward(v); 
        auto s_out = scalar_mlp->forward(torch::cat({s, (v_v * v_v).sum(1)}, 1)); 
        auto chunks = s_out.chunk(3, 1);
        auto delta_s = chunks[0] + (v_v * v_u).sum(1) * chunks[1]; 
        return {s + delta_s, v + v_u * chunks[2].unsqueeze(1)};
    }
};
TORCH_MODULE(PaiNNUpdate);

struct PaiNNModelImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    std::vector<PaiNNMessage> messages;
    std::vector<PaiNNUpdate> updates;
    torch::nn::Sequential readout{nullptr};
    int num_layers;

    PaiNNModelImpl(int num_embeddings, int dim, int layers) : num_layers(layers) {
        embedding = register_module("embedding", torch::nn::Embedding(num_embeddings, dim));
        for (int i = 0; i < layers; ++i) {
            messages.push_back(register_module("message_" + std::to_string(i), PaiNNMessage(dim)));
            updates.push_back(register_module("update_" + std::to_string(i), PaiNNUpdate(dim)));
        }
        readout = register_module("readout", torch::nn::Sequential(
            torch::nn::Linear(dim, dim / 2), torch::nn::SiLU(), torch::nn::Linear(dim / 2, 1)
        ));
    }

    torch::Tensor expansion_rbf(torch::Tensor d_ij) {
        auto centers = torch::linspace(0.0, 5.0, 20, d_ij.options());
        return torch::exp(-torch::pow(d_ij.unsqueeze(1) - centers, 2) / torch::pow(torch::full_like(centers, 0.5), 2));
    }

    torch::Tensor forward(PaiNNBatch& batch) {
        auto row = batch.edge_index[0], col = batch.edge_index[1]; 
        auto r_ij = batch.coordinates.index({col}) - batch.coordinates.index({row});
        auto d_ij = torch::norm(r_ij, 2, 1) + 1e-8; 

        torch::Tensor s = embedding->forward(batch.atomic_numbers);
        torch::Tensor v = torch::zeros({s.size(0), 3, s.size(1)}, s.options());

        auto r_ij_norm = r_ij / d_ij.unsqueeze(1);
        auto rbf = expansion_rbf(d_ij);

        for (int i = 0; i < num_layers; ++i) {
            auto [ds, dv] = messages[i]->forward(s, v, batch.edge_index, rbf, r_ij_norm);
            s = s + ds; v = v + dv;
            std::tie(s, v) = updates[i]->forward(s, v);
        }

        torch::Tensor atom_energies = readout->forward(s); 
        torch::Tensor pred_energy = torch::zeros({batch.energy_true.size(0), 1}, s.options());
        pred_energy.index_add_(0, batch.batch_indices, atom_energies);
        return pred_energy;
    }
};
TORCH_MODULE(PaiNNModel);

// =====================================================================
// 3. EARLY STOPPING & VALIDATION
// =====================================================================
struct EarlyStopping {
    int patience, counter;
    float best_loss;
    bool early_stop;
    std::string save_path;

    EarlyStopping(int p, std::string path) : 
        patience(p), counter(0), best_loss(std::numeric_limits<float>::infinity()), 
        early_stop(false), save_path(path) {}

    void check(float val_loss, PaiNNModel& model) {
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

struct ValidationMetrics {
    float combined_loss;
    float mae_energy;
    float mae_forces;
};

template <typename DatasetType>
ValidationMetrics validate_model(
    PaiNNModel& model, 
    DatasetType& val_dataset, 
    int batch_size,       
    float cutoff,         
    torch::Device device, 
    float force_weight = 30.0f) 
{
    model->eval();
    
    for (auto& param : model->parameters()) { param.set_requires_grad(false); }

    float total_mse_energy = 0.0f;
    float total_mse_forces = 0.0f;
    float total_mae_energy = 0.0f;
    float total_mae_forces = 0.0f;
    int num_batches = 0;

    for (size_t i = 0; i < val_dataset.size(); i += batch_size) {
        size_t end_idx = std::min(i + batch_size, val_dataset.size());
        std::vector<MoleculeFrame> batch_frames(val_dataset.begin() + i, val_dataset.begin() + end_idx);
        
        PaiNNBatch batch = collate_batch(batch_frames, cutoff, device);
        batch.coordinates.set_requires_grad(true);

        torch::Tensor E_pred = model->forward(batch);
        
        auto grads = torch::autograd::grad({E_pred}, {batch.coordinates}, {torch::ones_like(E_pred)}, false, false);
        torch::Tensor F_pred = -grads[0];

        {
            torch::NoGradGuard no_grad;
            
            // Metriche per lo Scheduler (MSE Loss)
            total_mse_energy += torch::mse_loss(E_pred, batch.energy_true).template item<float>();
            total_mse_forces += torch::mse_loss(F_pred, batch.forces_true).template item<float>();
            
            // Metriche per l'output grafico e di testo (MAE)
            total_mae_energy += torch::l1_loss(E_pred, batch.energy_true).template item<float>();
            total_mae_forces += torch::l1_loss(F_pred, batch.forces_true).template item<float>();
            
            num_batches++;
        }
    }

    for (auto& param : model->parameters()) { param.set_requires_grad(true); }

    ValidationMetrics metrics = {0.0f, 0.0f, 0.0f};
    if (num_batches > 0) {
        float avg_mse_energy = total_mse_energy / num_batches;
        float avg_mse_forces = total_mse_forces / num_batches;
        metrics.combined_loss = avg_mse_energy + (force_weight * avg_mse_forces);
        
        metrics.mae_energy = total_mae_energy / num_batches;
        metrics.mae_forces = total_mae_forces / num_batches;
    }

    return metrics;
}

// =====================================================================
// 4. MAIN LOOP
// =====================================================================
int main() {
    torch::Device device(torch::kCPU);

    if (torch::mps::is_available()) {
        device = torch::Device(torch::kMPS);
        std::cout << "\n[INFO] GPU Apple Silicon (Metal/MPS) rilevata ed attivata con successo!\n";
    } else {
        std::cout << "\n[INFO] Backend MPS non disponibile. Esecuzione su CPU.\n";
    }

    std::vector<MoleculeFrame> train_dataset, val_dataset;
    try {
        train_dataset = load_md17_binary("ethanol_train.bin");
        val_dataset   = load_md17_binary("ethanol_val.bin");
    } catch (const std::exception& e) {
        std::cerr << "Errore caricamento file binari: " << e.what() << "\n";
        return -1;
    }

    int batch_size = 32, max_epochs = 100;
    float cutoff = 5.0f, initial_lr = 5e-4;
    float force_weight = 30.0f;

    PaiNNModel model(100, 128, 3); 
    model->to(device);
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(initial_lr).weight_decay(1e-5));

    EarlyStopping early_stopping(10, "best_painn_etanolo.pt");
    
    std::mt19937 g(42); 
    float current_lr = initial_lr;
    int lr_patience = 4, lr_counter = 0;
    float best_val_loss = std::numeric_limits<float>::infinity();
    
    std::ofstream csv_file("training_metrics.csv");
    if (csv_file.is_open()) {
        csv_file << "Epoch,TrainLoss,ValLoss,MaeE,MaeF\n";
    } else {
        std::cerr << "Attenzione: Impossibile creare training_metrics.csv\n";
    }
    
    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        // --- TRAINING ---
        model->train();
        std::shuffle(train_dataset.begin(), train_dataset.end(), g);
        double train_mse_energy = 0.0, train_mse_forces = 0.0;
        double train_mae_energy = 0.0, train_mae_forces = 0.0;
        int train_batches = 0;

        for (size_t i = 0; i < train_dataset.size(); i += batch_size) {
            size_t end_idx = std::min(i + batch_size, train_dataset.size());
            std::vector<MoleculeFrame> batch_frames(train_dataset.begin() + i, train_dataset.begin() + end_idx);
            PaiNNBatch batch = collate_batch(batch_frames, cutoff, device);

            optimizer.zero_grad();
            batch.coordinates.set_requires_grad(true);

            torch::Tensor E_pred = model->forward(batch);
            auto grads = torch::autograd::grad({E_pred}, {batch.coordinates}, {torch::ones_like(E_pred)}, true, true);
            torch::Tensor F_pred = -grads[0];
            
            torch::Tensor loss_energy = torch::mse_loss(E_pred, batch.energy_true);
            torch::Tensor loss_forces = torch::mse_loss(F_pred, batch.forces_true);
            torch::Tensor total_loss = loss_energy + force_weight * loss_forces;

            total_loss.backward();
            optimizer.step();

            train_mse_energy += loss_energy.item<float>();
            train_mse_forces += loss_forces.item<float>();
            
            // Calcolo immediato delle MAE di Training (senza tracciare i gradienti)
            {
                torch::NoGradGuard no_grad;
                train_mae_energy += torch::l1_loss(E_pred, batch.energy_true).item<float>();
                train_mae_forces += torch::l1_loss(F_pred, batch.forces_true).item<float>();
            }
            
            train_batches++;
        }

        // --- VALIDATION ---
        ValidationMetrics val_metrics = validate_model(model, val_dataset, batch_size, cutoff, device, force_weight);
        
        float train_loss_tot = (train_mse_energy / train_batches) + force_weight * (train_mse_forces / train_batches);
        float avg_train_mae_e = train_mae_energy / train_batches;
        float avg_train_mae_f = train_mae_forces / train_batches;

        // --- STAMPA A SCHERMO DELLE METRICHE COMPRESE LE MAE ---
        std::cout << "\nEpoca [" << epoch << "/" << max_epochs << "] - LR: " << current_lr << "\n"
                  << "  [TRN] Loss: " << train_loss_tot 
                  << " | MAE Energia: " << avg_train_mae_e 
                  << " | MAE Forze: " << avg_train_mae_f << "\n"
                  << "  [VAL] Loss: " << val_metrics.combined_loss 
                  << " | MAE Energia: " << val_metrics.mae_energy 
                  << " | MAE Forze: " << val_metrics.mae_forces << "\n";
        // --- SCHEDULER (Reduce LR on Plateau) ---
        if (val_metrics.combined_loss < best_val_loss) {
            best_val_loss = val_metrics.combined_loss;
            lr_counter = 0;
        } else {
            lr_counter++;
            if (lr_counter >= lr_patience) {
                current_lr *= 0.5f;
                for (auto& options : optimizer.param_groups()) {
                    static_cast<torch::optim::AdamWOptions&>(options.options()).lr(current_lr);
                }
                std::cout << "  [Scheduler] Learning Rate abbassato a: " << current_lr << "\n";
                lr_counter = 0;
            }
        }
        
        // --- SCRITTURA METRICHE SU CSV ---
        if (csv_file.is_open()) {
            csv_file << epoch << ","
                     << train_loss_tot << ","
                     << val_metrics.combined_loss << ","
                     << val_metrics.mae_energy << ","
                     << val_metrics.mae_forces << "\n";
            
            csv_file.flush(); 
        }
        
        // --- EARLY STOPPING ---
        early_stopping.check(val_metrics.combined_loss, model);
        if (early_stopping.early_stop) {
            std::cout << "Stallo prolungato. Addestramento interrotto.\n";
            break;
        }
    }

    std::cout << "\nProcesso terminato con successo!\n";
    return 0; 
}
