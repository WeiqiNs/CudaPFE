include(CMakePackageConfigHelpers)

install(TARGETS cufe_core cufe_fe cufe_blst EXPORT CuFETargets)
install(DIRECTORY include/cufe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set(CUFE_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/CuFE)
install(EXPORT CuFETargets NAMESPACE CuFE:: DESTINATION ${CUFE_CMAKE_INSTALL_DIR})
configure_package_config_file(cmake/CuFEConfig.cmake.in CuFEConfig.cmake INSTALL_DESTINATION ${CUFE_CMAKE_INSTALL_DIR})
write_basic_package_version_file(CuFEConfigVersion.cmake COMPATIBILITY SameMinorVersion)
install(FILES ${PROJECT_BINARY_DIR}/CuFEConfig.cmake ${PROJECT_BINARY_DIR}/CuFEConfigVersion.cmake
        DESTINATION ${CUFE_CMAKE_INSTALL_DIR}
)
