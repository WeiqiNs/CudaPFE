add_executable(cufe_bench bench_core.cpp)
target_link_libraries(cufe_bench PRIVATE CuFE::core cufe_internal)
target_compile_definitions(cufe_bench PRIVATE CUFE_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

add_executable(cufe_bench_fe bench_fe.cpp)
target_include_directories(cufe_bench_fe PRIVATE ${PROJECT_SOURCE_DIR}/test/fe)
target_link_libraries(cufe_bench_fe PRIVATE CuFE::fe)
target_compile_definitions(cufe_bench_fe PRIVATE CUFE_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
