#include <torch/torch.h>
#include <iostream>
int main() {
    auto lin = torch::nn::Linear(10, 10);
    // torch::nn::utils::spectral_norm(lin); // does this exist in C++?
    std::cout << "OK" << std::endl;
    return 0;
}
