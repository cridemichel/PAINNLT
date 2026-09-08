cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(mlcg_train)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -std=c++17")

add_executable(train_painn train_painn.cpp)
target_link_libraries(train_painn "${TORCH_LIBRARIES}")
set_property(TARGET train_painn PROPERTY CXX_STANDARD 17)

# Su Apple il sorgente include ATen/mps/MPSAllocatorInterface.h, che tira dentro
# il runtime Objective-C: senza questo link i simboli objc_autoreleasePool*
# restano irrisolti.  Allineato a MLCG_Framework_v2/training/CMakeLists.txt.
if(APPLE)
    find_library(MLCG_OBJC_LIBRARY NAMES objc)
    if(NOT MLCG_OBJC_LIBRARY)
        message(FATAL_ERROR "Objective-C runtime library not found")
    endif()
    target_link_libraries(train_painn "${MLCG_OBJC_LIBRARY}")
endif()

add_executable(eval_parity eval_parity.cpp)
target_link_libraries(eval_parity "${TORCH_LIBRARIES}")
set_property(TARGET eval_parity PROPERTY CXX_STANDARD 17)
if(APPLE)
    target_link_libraries(eval_parity "${MLCG_OBJC_LIBRARY}")
endif()