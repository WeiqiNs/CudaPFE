cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures to build device code for")

project(
        CudaPFE
        VERSION 0.1
        DESCRIPTION "BLS12-381 pairings and pairing-based functional encryption on CPU and CUDA engines."
        LANGUAGES C CXX CUDA ASM
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
if (PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()

option(CUDAPFE_BUILD_TESTS "Build the CudaPFE test suite" ${PROJECT_IS_TOP_LEVEL})
option(CUDAPFE_BUILD_BENCH "Build the CudaPFE benchmark in bench/" OFF)
option(CUDAPFE_PTXAS_VERBOSE "Print ptxas register and spill reports for every kernel" OFF)
set(CUDAPFE_HOST_MILLER_BELOW 512 CACHE STRING "Gpu pair_segments runs everything on the host below this many pairs")
set(CUDAPFE_HOST_FINAL_EXP_BELOW 256 CACHE STRING "Gpu pair_segments runs final exponentiations on the host below this many segments")
set(CUDAPFE_HOST_POINTS_BELOW 256 CACHE STRING "Gpu point operations run on the host below this many results or msm terms")

include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>)
if (CUDAPFE_PTXAS_VERBOSE)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif ()

include(cmake/CudaPFEDependencies.cmake)

add_library(cudapfe_core STATIC
        src/curve/point.cu
        src/dlog/dlog.cu
        src/field/random.cpp
        src/field/zp.cu
        src/linalg/matrix.cu
        src/pairing/gt.cu
        src/pairing/multi_pair.cu
        src/pairing/pair.cu
        src/support/runtime.cu
        src/vec/vec.cu
)
add_library(CudaPFE::core ALIAS cudapfe_core)
target_compile_features(cudapfe_core PUBLIC cxx_std_20)
target_include_directories(cudapfe_core
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(cudapfe_core
        PRIVATE $<BUILD_INTERFACE:cudapfe_internal> OpenMP::OpenMP_CXX
        PUBLIC cudapfe_blst CUDA::cudart
)
set_target_properties(cudapfe_core PROPERTIES EXPORT_NAME core POSITION_INDEPENDENT_CODE ON)

add_library(cudapfe_fe INTERFACE)
add_library(CudaPFE::fe ALIAS cudapfe_fe)
target_link_libraries(cudapfe_fe INTERFACE cudapfe_core)
set_target_properties(cudapfe_fe PROPERTIES EXPORT_NAME fe)

include(cmake/CudaPFEInstall.cmake)

if (CUDAPFE_BUILD_BENCH)
    add_subdirectory(bench)
endif ()

if (CUDAPFE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()
