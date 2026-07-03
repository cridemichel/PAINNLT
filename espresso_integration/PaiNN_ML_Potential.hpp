#pragma once

#include <memory>
#include <string>

// ESPResSo includes
#include "Particle.hpp"
#include "cells.hpp"
#include "nonbonded_interactions/VerletCriterion.hpp"

// PaiNN includes
#include "PaiNN_Architecture.hpp"
#include <torch/torch.h>

class PaiNN_ML_Potential {
public:
    PaiNN_ML_Potential(const std::string& model_path, int num_species, int hidden_channels, int n_layers, int num_rbf, double cutoff, const std::string& device_str = "auto");

    // Evaluates the ML potential and adds forces to particles
    void calculate_forces(CellStructure& cell_structure, const VerletCriterion<>& verlet_criterion);

    double get_cutoff() const { return m_cutoff; }

    // Ritorna l'ultima energia potenziale calcolata dal modello
    double get_last_energy() const { return m_last_energy; }

private:
    PaiNNModel model{nullptr};
    double m_cutoff;
    double m_last_energy = 0.0;
    torch::Device m_device{torch::kCPU};
};

// Global instance to be used in integrate.cpp or forces.cpp
extern std::shared_ptr<PaiNN_ML_Potential> global_painn_potential;
