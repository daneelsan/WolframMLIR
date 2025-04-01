llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(WOLFRAM_TEST_DEPENDS
        FileCheck count not
        #wolfram-capi-test
        wolfram-opt
        #wolfram-translate
        WolframPlugin
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND WOLFRAM_TEST_DEPENDS StandalonePythonModules)
endif()

add_lit_testsuite(check-wolfram "Running the wolfram regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${WOLFRAM_TEST_DEPENDS}
        )
set_target_properties(check-wolfram PROPERTIES FOLDER "Tests")

add_lit_testsuites(WOLFRAM ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${WOLFRAM_TEST_DEPENDS})

#add_subdirectory(CAPI)
