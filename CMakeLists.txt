cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(painn_framework CXX)

# --- AGGIUNGI QUESTA RIGA ---
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Richiedi C++17 per LibTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cerca LibTorch nel sistema
find_package(Torch REQUIRED)

# Definisci l'eseguibile a partire dal file unico
add_executable(painn_app painn.cpp)

# Collega le librerie di PyTorch
target_link_libraries(painn_app "${TORCH_LIBRARIES}")

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(painn_framework CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

