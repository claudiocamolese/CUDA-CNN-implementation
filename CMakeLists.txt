cmake_minimum_required(VERSION 3.22)
project(CUDA_CNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# --- BASELINE CUDA ---
# =========================
set(BASELINE_SRC
    src/baseline_cuda/main.cu
    src/utils/load_data.cu
    src/baseline_cuda/kernels/forward.cu
    src/baseline_cuda/kernels/backward.cu
)

add_executable(cnn_baseline ${BASELINE_SRC})

set_target_properties(cnn_baseline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(cnn_baseline PRIVATE
    ${CMAKE_SOURCE_DIR}      # config.h
    ${CMAKE_SOURCE_DIR}/src  # utils/, baseline_cuda/, optimized_cuda/
    src/baseline_cuda
    src/baseline_cuda/kernels
)

# =========================
# --- OPTIMIZED CUDA ---
# =========================
set(OPTIMIZED_SRC
    src/optimized_cuda/main.cu
    src/utils/load_data.cu
    src/optimized_cuda/kernels/forward.cu
    src/optimized_cuda/kernels/backward.cu
)

add_executable(cnn_optimized ${OPTIMIZED_SRC})

set_target_properties(cnn_optimized PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(cnn_optimized PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    src/optimized_cuda
    src/optimized_cuda/kernels
)
