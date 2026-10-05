include(FetchContent)

set(CUFE_BLST_GIT_TAG "54e6e55674722fc2797ebb4bbb71b26d881eb4b8" CACHE STRING "blst branch, tag or commit to build")

FetchContent_Declare(blst
        GIT_REPOSITORY https://github.com/supranational/blst.git
        GIT_TAG ${CUFE_BLST_GIT_TAG}
        SOURCE_SUBDIR cufe-do-not-add-subdirectory
)
FetchContent_MakeAvailable(blst)

execute_process(
        COMMAND git -C ${blst_SOURCE_DIR} rev-parse HEAD
        OUTPUT_VARIABLE CUFE_BLST_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)
message(STATUS "LibCuFE: blst ${CUFE_BLST_GIT_TAG} resolved to '${CUFE_BLST_COMMIT}'")

add_library(cufe_blst STATIC ${blst_SOURCE_DIR}/src/server.c ${blst_SOURCE_DIR}/build/assembly.S)
target_compile_definitions(cufe_blst PUBLIC __BLST_PORTABLE__)
target_compile_options(cufe_blst PRIVATE $<$<COMPILE_LANGUAGE:C>:-fno-builtin>)
target_include_directories(cufe_blst PUBLIC $<BUILD_INTERFACE:${blst_SOURCE_DIR}/bindings>)
set_target_properties(cufe_blst PROPERTIES EXPORT_NAME blst POSITION_INDEPENDENT_CODE ON)

add_library(cufe_internal INTERFACE)
target_include_directories(cufe_internal INTERFACE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(cufe_internal SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/sppark ${blst_SOURCE_DIR}/src)
target_link_libraries(cufe_internal INTERFACE cufe_blst CUDA::cudart)
target_compile_definitions(cufe_internal INTERFACE
        CUFE_HOST_MILLER_BELOW=${CUFE_HOST_MILLER_BELOW}
        CUFE_HOST_FINAL_EXP_BELOW=${CUFE_HOST_FINAL_EXP_BELOW}
)
