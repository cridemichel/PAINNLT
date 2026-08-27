cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mlcg_train LANGUAGES CXX)

# PyTorch's TorchConfig.cmake uses find_library() without NO_DEFAULT_PATH.
# When pip/venv Torch and Homebrew LibTorch coexist, that can mix the config
# and headers from one installation with libtorch/c10 from the other one.
# Resolve and pin the library root before loading TorchConfig.cmake.
set(MLCG_TORCH_ROOT "" CACHE PATH
    "Root of the single LibTorch/Python-Torch distribution used for this build")

if(NOT MLCG_TORCH_ROOT AND DEFINED Torch_DIR)
    get_filename_component(MLCG_TORCH_ROOT "${Torch_DIR}/../../.." ABSOLUTE)
endif()

if(NOT MLCG_TORCH_ROOT)
    foreach(prefix IN LISTS CMAKE_PREFIX_PATH)
        if(EXISTS "${prefix}/Torch/TorchConfig.cmake")
            get_filename_component(MLCG_TORCH_ROOT "${prefix}/../.." ABSOLUTE)
            set(Torch_DIR "${prefix}/Torch" CACHE PATH "Torch CMake package" FORCE)
            break()
        elseif(EXISTS "${prefix}/share/cmake/Torch/TorchConfig.cmake")
            get_filename_component(MLCG_TORCH_ROOT "${prefix}" ABSOLUTE)
            set(Torch_DIR "${prefix}/share/cmake/Torch" CACHE PATH
                "Torch CMake package" FORCE)
            break()
        endif()
    endforeach()
endif()

if(MLCG_TORCH_ROOT)
    get_filename_component(MLCG_TORCH_ROOT "${MLCG_TORCH_ROOT}" REALPATH)
    if(EXISTS "${MLCG_TORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
        set(Torch_DIR "${MLCG_TORCH_ROOT}/share/cmake/Torch" CACHE PATH
            "Torch CMake package" FORCE)
    endif()
    if(NOT EXISTS "${MLCG_TORCH_ROOT}/share/cmake/Caffe2/Caffe2Config.cmake")
        message(FATAL_ERROR
            "MLCG_TORCH_ROOT does not contain Caffe2Config.cmake: ${MLCG_TORCH_ROOT}")
    endif()
    set(Caffe2_DIR "${MLCG_TORCH_ROOT}/share/cmake/Caffe2" CACHE PATH
        "Caffe2 CMake package from the selected Torch distribution" FORCE)
    unset(MLCG_TORCH_LIBRARY CACHE)
    unset(MLCG_C10_LIBRARY CACHE)
    find_library(MLCG_TORCH_LIBRARY NAMES torch
        PATHS "${MLCG_TORCH_ROOT}/lib" NO_DEFAULT_PATH)
    find_library(MLCG_C10_LIBRARY NAMES c10
        PATHS "${MLCG_TORCH_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT MLCG_TORCH_LIBRARY OR NOT MLCG_C10_LIBRARY)
        message(FATAL_ERROR
            "MLCG_TORCH_ROOT does not contain libtorch and libc10: ${MLCG_TORCH_ROOT}")
    endif()
    set(TORCH_LIBRARY "${MLCG_TORCH_LIBRARY}" CACHE FILEPATH
        "Pinned libtorch library" FORCE)
    set(c10_LIBRARY "${MLCG_C10_LIBRARY}" CACHE FILEPATH
        "Pinned c10 library" FORCE)
endif()

find_package(Torch REQUIRED)

if(NOT MLCG_TORCH_ROOT AND DEFINED TORCH_INSTALL_PREFIX)
    get_filename_component(MLCG_TORCH_ROOT "${TORCH_INSTALL_PREFIX}" REALPATH)
endif()

if(MLCG_TORCH_ROOT)
    foreach(torch_library_var IN ITEMS TORCH_LIBRARY c10_LIBRARY)
        get_filename_component(torch_library_real "${${torch_library_var}}" REALPATH)
        string(FIND "${torch_library_real}" "${MLCG_TORCH_ROOT}/" root_position)
        if(NOT root_position EQUAL 0)
            message(FATAL_ERROR
                "Mixed LibTorch installations: ${torch_library_var}=${torch_library_real}, "
                "expected a library below ${MLCG_TORCH_ROOT}. Reconfigure with --fresh.")
        endif()
    endforeach()
    message(STATUS "Using a single LibTorch root: ${MLCG_TORCH_ROOT}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -std=c++17")

add_executable(train_painn train_painn.cpp)
target_link_libraries(train_painn PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET train_painn PROPERTY CXX_STANDARD 17)

if(APPLE)
    find_library(MLCG_OBJC_LIBRARY NAMES objc)
    if(NOT MLCG_OBJC_LIBRARY)
        message(FATAL_ERROR "Objective-C runtime library not found")
    endif()
    target_link_libraries(train_painn PRIVATE "${MLCG_OBJC_LIBRARY}")
endif()

add_executable(eval_parity eval_parity.cpp)
target_link_libraries(eval_parity PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET eval_parity PROPERTY CXX_STANDARD 17)
