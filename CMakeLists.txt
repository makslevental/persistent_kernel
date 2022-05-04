cmake_minimum_required(VERSION 3.22)
project(persistent_kernel CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(persistent_kernel main.cu)

set_target_properties(persistent_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(persistent_kernel ${CUDA_LIBRARIES} cuda::cuda CUDA::cuda_driver)

# Add the cuda imported target
if (NOT TARGET cuda::cuda)
    add_library(cuda::cuda INTERFACE IMPORTED)
endif ()

set_property(TARGET cuda::cuda
        PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARIES})

set_property(TARGET cuda::cuda
        PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})
