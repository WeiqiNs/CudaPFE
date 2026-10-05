include(CMakePackageConfigHelpers)

install(TARGETS cufe_core cufe_blst EXPORT CuFETargets)
install(DIRECTORY include/cufe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set(CUFE_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/CuFE)
install(EXPORT CuFETargets NAMESPACE CuFE:: DESTINATION ${CUFE_CMAKE_INSTALL_DIR})
configure_package_config_file(cmake/CuFEConfig.cmake.in CuFEConfig.cmake INSTALL_DESTINATION ${CUFE_CMAKE_INSTALL_DIR})
write_basic_package_version_file(CuFEConfigVersion.cmake COMPATIBILITY SameMinorVersion)
install(FILES ${PROJECT_BINARY_DIR}/CuFEConfig.cmake ${PROJECT_BINARY_DIR}/CuFEConfigVersion.cmake
        DESTINATION ${CUFE_CMAKE_INSTALL_DIR}
)
install(FILES LICENSE.md NOTICE DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(FILES ${blst_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR}/blst)
install(FILES third_party/sppark/LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR}/sppark)
