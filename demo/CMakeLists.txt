cmake_minimum_required(VERSION 3.25)
project(CudaPFEDemo LANGUAGES CXX)

find_package(CudaPFE REQUIRED)

add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE CudaPFE::core)
