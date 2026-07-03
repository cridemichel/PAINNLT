# Integration Guide for PaiNN in ESPResSo

This folder contains all the custom source files that let you use your **PaiNN** model (implemented in C++ with **libtorch**) as a native potential in **ESPResSo**.

The idea is that you can start from a *clean* ESPResSo source tree and apply this patch to obtain a high‑performance coarse‑grained (CG) potential.

## Files Included

| File | Description |
|------|-------------|
| **`PaiNN_Architecture.hpp`** | The original PaiNN model, extended with a `forward_with_rij` method that directly accepts distance vectors so it can respect ESPResSo’s Minimum Image Convention (PBC) natively. |
| **`PaiNN_ML_Potential.hpp`** / **`.cpp`** | Bridge classes that connect the model to ESPResSo. They gather particle data, build the neighbour list, call the forward pass for the energy, and return the gradients (forces) to the particles. |
| **`painn.pyx`** | Cython interface that lets you instantiate and call the potential from Python just like any other ESPResSo potential. |

---

## How to Apply the Patch to ESPResSo

Assuming you have cloned ESPResSo and you are in the project root (`espresso/`), follow these exact steps:

### 1. Copy the Files

```bash
# Copy the headers and the cpp source into ESPResSo’s core
cp espresso_integration/PaiNN_Architecture.hpp   espresso/src/core/nonbonded_interactions/
cp espresso_integration/PaiNN_ML_Potential.hpp   espresso/src/core/nonbonded_interactions/
cp espresso_integration/PaiNN_ML_Potential.cpp   espresso/src/core/nonbonded_interactions/

# Copy the Python (Cython) interface
cp espresso_integration/painn.pyx               espresso/src/python/espressomd/
```

### 2. Edit `espresso/src/core/CMakeLists.txt`

Open `espresso/src/core/CMakeLists.txt` and make two changes:

1. **Before the `target_link_libraries(espresso_core …)` line** (usually around line 77), add the Torch package:

```cmake
find_package(Torch REQUIRED)
```
2. **Inside the `target_link_libraries(espresso_core PUBLIC …)` block**, add `${TORCH_LIBRARIES}`.
3. **Add a compile definition** so the code knows we are building with PaiNN:

```cmake
target_compile_definitions(espresso_core PUBLIC ESPRESSO_PAINN)
```

### 3. Edit `espresso/src/core/nonbonded_interactions/CMakeLists.txt`

Add the new source file to the list of compiled files (inside `target_sources(espresso_core PRIVATE …)`):

```cmake
${CMAKE_CURRENT_SOURCE_DIR}/PaiNN_ML_Potential.cpp
```

### 4. Edit `espresso/src/core/forces.cpp`

Open `espresso/src/core/forces.cpp` and apply two modifications:

**A.** At the top of the file, together with the other `#include`s, add:

```cpp
#ifdef ESPRESSO_PAINN
#include "nonbonded_interactions/PaiNN_ML_Potential.hpp"
#endif
```

**B.** Locate the function `System::calculate_forces()`.  Just before the function ends—right after the block that finishes the short‑range (cabana) forces—insert the call to our global PaiNN potential.

Find the lines (around 370‑375) that look like this:

```cpp
#ifdef ESPRESSO_CALIPER
  CALI_MARK_END("cabana_short_range");
#endif
```

Immediately **after** that block, add:

```cpp
#ifdef ESPRESSO_PAINN
  if (global_painn_potential) {
    global_painn_potential->calculate_forces(*cell_structure, verlet_criterion);
  }
#endif
```

*(Note: Cython files under `espressomd/` are automatically discovered by CMake via a `*.pyx` glob, so you do not need to add `painn.pyx` manually to the Python CMake configuration.)*

---

## Compilation

Once the patch is applied, compile ESPResSo while pointing CMake to your **libtorch** installation:

```bash
cd espresso
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/absolute/path/to/your/libtorch
make -j4            # Adjust the parallelism level to your CPU
```

---

## Running & Testing

Inside this `espresso_integration` folder you will find two ready‑to‑run Python scripts that validate the implementation.

### 5.1 Basic Test (`test_espresso_painn.py`)

Run it with the ESPResSo Python executable:

```bash
./espresso/build/pypresso espresso_integration/test_espresso_painn.py
```

The script creates **9 atoms** at random positions, activates the C++ PaiNN potential, and performs **0 integration steps** just to retrieve the predicted forces.

### 5.2 Quantitative Validation (`validate_single_point.py`)

Use this script to compare the forces returned by ESPResSo with the *reference* forces from the original dataset (useful for computing the MAE).

```bash
./espresso/build/pypresso espresso_integration/validate_single_point.py
```

*(Make sure the files `ethanol_val.bin` and the `*.pt` weights are present in the directory from which you launch the script.)*

---

## **⚠️ Important – Verlet Lists**

ESPResSo builds neighbour (Verlet) lists **only** if there is at least one non‑bonded interaction registered. Our potential is a global C++ potential, so ESPResSo does **not** know about the 5 Å cutoff we use in PaiNN.

To force ESPResSo to build the correct neighbour list, you must always define a **dummy (zero‑energy) Lennard‑Jones interaction** with a cutoff equal to or larger than the model’s cutoff in your Python script, e.g.:

```python
# Create a zero‑energy interaction with cutoff = 5.0
# This forces ESPResSo to search for neighbours up to that distance.
for i in range(10):                 # for each atom type
    for j in range(i, 10):
        system.non_bonded_inter[i, j].lennard_jones.set_params(
            epsilon=0.0, sigma=1.0, cutoff=5.0, shift=0.0)
```

Also remember that the minimal box length must satisfy the condition `box_l / 2 > cutoff + skin`.  With `cutoff = 5.0` and `skin = 0.4`, the box should be at least **10.9 × 10.9 × 10.9**.

---

That’s it! Follow the steps above to integrate your PaiNN model into ESPResSo, compile, and run the provided test scripts to verify everything works correctly. If you encounter any issues or need further customization, just let me know.

