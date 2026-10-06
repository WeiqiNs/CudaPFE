include(CMakePackageConfigHelpers)

install(TARGETS cudapfe_core cudapfe_fe cudapfe_blst EXPORT CudaPFETargets)
install(DIRECTORY include/cudapfe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set(CUDAPFE_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/CudaPFE)
install(EXPORT CudaPFETargets NAMESPACE CudaPFE:: DESTINATION ${CUDAPFE_CMAKE_INSTALL_DIR})
configure_package_config_file(cmake/CudaPFEConfig.cmake.in CudaPFEConfig.cmake INSTALL_DESTINATION ${CUDAPFE_CMAKE_INSTALL_DIR})
write_basic_package_version_file(CudaPFEConfigVersion.cmake COMPATIBILITY SameMinorVersion)
install(FILES ${PROJECT_BINARY_DIR}/CudaPFEConfig.cmake ${PROJECT_BINARY_DIR}/CudaPFEConfigVersion.cmake
        DESTINATION ${CUDAPFE_CMAKE_INSTALL_DIR}
)
