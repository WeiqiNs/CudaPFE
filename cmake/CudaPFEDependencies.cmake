include(FetchContent)

set(CUDAPFE_BLST_GIT_TAG "54e6e55674722fc2797ebb4bbb71b26d881eb4b8" CACHE STRING "blst branch, tag or commit to build")

FetchContent_Declare(blst
        GIT_REPOSITORY https://github.com/supranational/blst.git
        GIT_TAG ${CUDAPFE_BLST_GIT_TAG}
        SOURCE_SUBDIR cudapfe-do-not-add-subdirectory
)
FetchContent_MakeAvailable(blst)

execute_process(
        COMMAND git -C ${blst_SOURCE_DIR} rev-parse HEAD
        OUTPUT_VARIABLE CUDAPFE_BLST_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)
message(STATUS "CudaPFE: blst ${CUDAPFE_BLST_GIT_TAG} resolved to '${CUDAPFE_BLST_COMMIT}'")

add_library(cudapfe_blst STATIC ${blst_SOURCE_DIR}/src/server.c ${blst_SOURCE_DIR}/build/assembly.S)
target_compile_definitions(cudapfe_blst PUBLIC __BLST_PORTABLE__)
target_compile_options(cudapfe_blst PRIVATE $<$<COMPILE_LANGUAGE:C>:-fno-builtin>)
target_include_directories(cudapfe_blst PUBLIC $<BUILD_INTERFACE:${blst_SOURCE_DIR}/bindings>)
set_target_properties(cudapfe_blst PROPERTIES EXPORT_NAME blst POSITION_INDEPENDENT_CODE ON)

add_library(cudapfe_internal INTERFACE)
target_include_directories(cudapfe_internal INTERFACE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(cudapfe_internal SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/third_party/sppark ${blst_SOURCE_DIR}/src)
target_link_libraries(cudapfe_internal INTERFACE cudapfe_blst CUDA::cudart)
target_compile_definitions(cudapfe_internal INTERFACE
        CUDAPFE_HOST_MILLER_BELOW=${CUDAPFE_HOST_MILLER_BELOW}
        CUDAPFE_HOST_FINAL_EXP_BELOW=${CUDAPFE_HOST_FINAL_EXP_BELOW}
)
