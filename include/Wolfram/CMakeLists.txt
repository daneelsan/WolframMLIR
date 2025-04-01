add_mlir_dialect(WolframOps wolfram)
add_mlir_doc(WolframDialect WolframDialect Wolfram/ -gen-dialect-doc)
add_mlir_doc(WolframOps WolframOps Wolfram/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS WolframPasses.td)
mlir_tablegen(WolframPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRWolframPassesIncGen)
