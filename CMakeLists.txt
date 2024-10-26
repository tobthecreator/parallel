cmake_minimum_required(VERSION 3.10)
project(CUDAProject CUDA CXX)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust based on your GPU

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add includes
include_directories(${CUDA_INCLUDE_DIRS})

# Add executable
add_executable(cuda_test src/main.cu)

# Set CUDA specific flags
set_target_properties(cuda_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)