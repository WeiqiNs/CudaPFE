cmake_minimum_required(VERSION 3.25)

set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures to build device code for")

project(
        LibCuFE
        VERSION 0.1
        DESCRIPTION "BLS12-381 pairings and function-hiding inner-product encryption on CPU and CUDA engines."
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

option(CUFE_BUILD_TESTS "Build the LibCuFE test suite" ${PROJECT_IS_TOP_LEVEL})
option(CUFE_PTXAS_VERBOSE "Print ptxas register and spill reports for every kernel" OFF)
set(CUFE_HOST_MILLER_BELOW 64 CACHE STRING "Gpu pair_segments runs everything on the host below this many pairs")
set(CUFE_HOST_FINAL_EXP_BELOW 32 CACHE STRING "Gpu pair_segments runs final exponentiations on the host below this many segments")

include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-Werror=cross-execution-space-call>)
if (CUFE_PTXAS_VERBOSE)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif ()

include(cmake/CuFEDependencies.cmake)

add_library(cufe_core STATIC
        src/curve/point.cu
        src/field/random.cpp
        src/field/zp.cu
        src/pairing/gt.cu
        src/pairing/multi_pair.cu
        src/pairing/pair.cu
        src/support/runtime.cu
        src/vec/vec.cu
)
add_library(CuFE::core ALIAS cufe_core)
target_compile_features(cufe_core PUBLIC cxx_std_20)
target_include_directories(cufe_core
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(cufe_core PRIVATE $<BUILD_INTERFACE:cufe_internal> PUBLIC cufe_blst CUDA::cudart)
set_target_properties(cufe_core PROPERTIES EXPORT_NAME core POSITION_INDEPENDENT_CODE ON)

include(cmake/CuFEInstall.cmake)

if (CUFE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()
