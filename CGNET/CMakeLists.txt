cmake_minimum_required(VERSION 3.14)
project(CG_SchNet_Train)

set(CMAKE_CXX_STANDARD 17)

# Cerca LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Aggiungi l'eseguibile
add_executable(cg_schnet_train cg_schnet_train.cpp)

# Linka LibTorch
target_link_libraries(cg_schnet_train "${TORCH_LIBRARIES}")
set_property(TARGET cg_schnet_train PROPERTY CXX_STANDARD 17)
