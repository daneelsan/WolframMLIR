add_mlir_dialect_library(MLIRWolfram
        WolframTypes.cpp
        WolframDialect.cpp
        WolframOps.cpp
        WolframPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Wolfram
        #${WOLFRAM_BINARY_DIR}/include/Wolfram  # Include the build directory for generated .inc files

        DEPENDS
        MLIRWolframOpsIncGen
        MLIRWolframPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
