add_llvm_library(WolframPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        wolfram-plugin.cpp

        DEPENDS
        MLIRWolfram
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRWolfram
        )
target_include_directories(
        WolframPlugin
        PRIVATE
        "${WOLFRAM_BINARY_DIR}/include"
)
