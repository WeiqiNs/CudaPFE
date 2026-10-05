add_executable(cufe_bench bench_core.cpp)
target_link_libraries(cufe_bench PRIVATE CuFE::core cufe_internal)
target_compile_definitions(cufe_bench PRIVATE CUFE_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
