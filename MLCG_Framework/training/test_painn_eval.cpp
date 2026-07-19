#include "PaiNN_Architecture.hpp"
#include <torch/torch.h>
#include <iostream>

int main() {
    try {
        auto model = std::make_shared<PaiNNModel>(8, 64, 3, 50, 1.0);
        torch::load(model, "tel22_model.pt");
        std::cout << "Model loaded successfully." << std::endl;
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
