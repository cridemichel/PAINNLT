cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(mlcg_train)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -std=c++17")

add_executable(train_painn train_painn.cpp)
target_link_libraries(train_painn "${TORCH_LIBRARIES}")
set_property(TARGET train_painn PROPERTY CXX_STANDARD 17)
