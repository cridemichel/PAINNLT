import re

with open("PaiNN_Architecture.hpp", "r") as f:
    content = f.read()

# Restore bias to filter_mlp
content = content.replace(
    'filter_mlp = register_module("filter_mlp", torch::nn::Linear(torch::nn::LinearOptions(num_rbf, dim * 3).bias(false)));',
    'filter_mlp = register_module("filter_mlp", torch::nn::Linear(num_rbf, dim * 3));'
)

# Update PaiNNMessageImpl forward signature and apply cos_cutoff
old_forward_msg = 'std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor s, torch::Tensor v, torch::Tensor edge_index, torch::Tensor rbf, torch::Tensor r_ij_norm) {'
new_forward_msg = 'std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor s, torch::Tensor v, torch::Tensor edge_index, torch::Tensor rbf, torch::Tensor r_ij_norm, torch::Tensor cos_cutoff) {'
content = content.replace(old_forward_msg, new_forward_msg)

old_w_calc = 'auto w = filter_mlp->forward(rbf); \n        auto interaction = scalar_mlp->forward(s.index({row})) * w;'
new_w_calc = 'auto w = filter_mlp->forward(rbf); \n        w = w * cos_cutoff.unsqueeze(1); // SOTA: applica il cutoff DOPO il layer lineare per azzerare il bias!\n        auto interaction = scalar_mlp->forward(s.index({row})) * w;'
content = content.replace(old_w_calc, new_w_calc)

# Update expansion_rbf to return pair
old_expansion = 'torch::Tensor expansion_rbf(torch::Tensor d_ij)'
new_expansion = 'std::pair<torch::Tensor, torch::Tensor> expansion_rbf(torch::Tensor d_ij)'
content = content.replace(old_expansion, new_expansion)

old_expansion_return = 'return rbf * cos_cutoff.unsqueeze(1);'
new_expansion_return = 'return {rbf * cos_cutoff.unsqueeze(1), cos_cutoff};'
content = content.replace(old_expansion_return, new_expansion_return)

# Update forward and forward_with_rij
content = content.replace('auto rbf = expansion_rbf(d_ij);', 'auto rbf_pair = expansion_rbf(d_ij);\n        auto rbf = rbf_pair.first;\n        auto cos_cutoff = rbf_pair.second;')
content = content.replace('messages[i]->forward(s, v, batch.edge_index, rbf, r_ij_norm)', 'messages[i]->forward(s, v, batch.edge_index, rbf, r_ij_norm, cos_cutoff)')
content = content.replace('messages[i]->forward(s, v, edge_index, rbf, r_ij_norm)', 'messages[i]->forward(s, v, edge_index, rbf, r_ij_norm, cos_cutoff)')

with open("PaiNN_Architecture.hpp", "w") as f:
    f.write(content)
