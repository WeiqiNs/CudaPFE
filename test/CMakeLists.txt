find_package(GTest QUIET)
if (NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.18.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif ()
include(GoogleTest)

function(cufe_add_test name source)
    cmake_parse_arguments(PARSE_ARGV 2 arg "INTERNAL" "" "")
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} PRIVATE GTest::gtest_main CuFE::core)
    if (arg_INTERNAL)
        target_link_libraries(${name} PRIVATE cufe_internal)
    endif ()
    gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST PROPERTIES SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" TIMEOUT 300)
endfunction()

cufe_add_test(test_toolchain internal/test_toolchain.cu INTERNAL)
cufe_add_test(test_field internal/test_field.cu INTERNAL)
cufe_add_test(test_tower internal/test_tower.cu INTERNAL)
cufe_add_test(test_curve internal/test_curve.cu INTERNAL)
