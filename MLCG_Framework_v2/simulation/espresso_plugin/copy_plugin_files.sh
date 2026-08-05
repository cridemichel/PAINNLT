#!/bin/bash
# Copy the headers and the cpp source into ESPResSo’s core
cp -f espresso_plugin/PaiNN_Architecture.hpp  ../espresso/src/core/nonbonded_interactions/
cp -f espresso_plugin/PaiNN_ML_Potential.hpp  ../espresso/src/core/nonbonded_interactions/
cp -f espresso_plugin/PaiNN_ML_Potential.cpp  ../espresso/src/core/nonbonded_interactions/
cp -f espresso_plugin/painn.pyx               ../espresso/src/python/espressomd/
