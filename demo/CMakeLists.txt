cmake_minimum_required(VERSION 3.25)
project(LibCuFEDemo LANGUAGES CXX)

find_package(CuFE REQUIRED)

add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE CuFE::core)
