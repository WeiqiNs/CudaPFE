add_executable(cudapfe_bench bench_core.cpp)
target_link_libraries(cudapfe_bench PRIVATE CudaPFE::core cudapfe_internal)
target_compile_definitions(cudapfe_bench PRIVATE CUDAPFE_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

add_executable(cudapfe_bench_fe bench_fe.cpp)
target_include_directories(cudapfe_bench_fe PRIVATE ${PROJECT_SOURCE_DIR}/test/fe)
target_link_libraries(cudapfe_bench_fe PRIVATE CudaPFE::fe)
target_compile_definitions(cudapfe_bench_fe PRIVATE CUDAPFE_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
