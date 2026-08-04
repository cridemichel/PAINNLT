import re
import os

with open("train_painn.cpp", "r") as f:
    code = f.read()

match = re.search(r"int\s+main\s*\(\s*int\s+argc,\s*char\s*\*\s*argv\[\]\s*\)", code)
if match:
    main_idx = match.start()
else:
    print("Main non trovato!")
    exit(1)

header = code[:main_idx]

main_func = """int main(int argc, char** argv) {
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
        std::cerr << "Config non trovato: " << config_file << "\\n";
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
        std::cerr << "[ERROR] " << e.what() << "\\n";
        return 1;
    }
    std::cout << "Excluded 1-2/1-3 molecule pairs: "
              << excluded_molecule_pairs.size() << "\\n";
    
    std::vector<CGFrame> dataset = read_cg_dataset(dataset_file);
    if (dataset.empty()) return 1;
    
    // Per avere un campione significativo senza metterci ore
    // saltiamo parte del dataset di train e prediamo l'intero validation set
    size_t val_size = dataset.size() * 0.2;
    size_t train_size = dataset.size() - val_size;
    std::vector<CGFrame> val_dataset(dataset.begin() + train_size, dataset.end());
    
    torch::Device device = torch::kCPU;
#ifdef __APPLE__
    device = torch::Device(torch::kMPS);
#elif defined(__linux__)
    if (torch::cuda::is_available()) device = torch::Device(torch::kCUDA);
#endif
    std::cout << "Evaluating on " << (device.type() == torch::kMPS ? "MPS" : (device.type() == torch::kCUDA ? "CUDA" : "CPU")) << "...\\n";
    
    PaiNNModel model(
        config["num_species"], config["hidden_channels"], config["n_layers"], 
        config["num_rbf"], cutoff, config.value("apply_envelope", true), 
        config.value("use_bias", false), config.value("toxvaerd_alpha", 0.1)
    );
    
    torch::load(model, model_file);
    model->to(device);
    model->eval();
    
    std::ofstream out_csv("parity_forces.csv");
    out_csv << "F_target_x,F_target_y,F_target_z,F_pred_x,F_pred_y,F_pred_z\\n";
    
    size_t batch_size = 5;
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
                        << pred_f_acc[m][0] << "," << pred_f_acc[m][1] << "," << pred_f_acc[m][2] << "\\n";
            }
            val_batch_frames.clear();
            
            progress++;
            if(progress % 10 == 0) std::cout << "Evaluated " << i << "/" << val_dataset.size() << " frames...\\n";
        }
    }
    out_csv.close();
    std::cout << "Done! parity_forces.csv written.\\n";
    return 0;
}
"""

with open("eval_parity.cpp", "w") as f:
    f.write(header + main_func)

# Aggiungiamo al CMakeLists
with open("CMakeLists.txt", "r") as f:
    cm = f.read()

if "eval_parity" not in cm:
    with open("CMakeLists.txt", "a") as f:
        f.write("\\nadd_executable(eval_parity eval_parity.cpp)\\n")
        f.write('target_link_libraries(eval_parity "${TORCH_LIBRARIES}")\\n')
        f.write("set_property(TARGET eval_parity PROPERTY CXX_STANDARD 17)\\n")
