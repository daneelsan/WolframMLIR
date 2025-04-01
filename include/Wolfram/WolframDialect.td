// build/bin/mlir-tblgen -gen-dialect-decls ~/projects/thesis/wolfram-mlir/tablegen/WolframDialect.td ~/projects/thesis/wolfram-mlir/include/Wolfram/Ops.td -I mlir/include

#ifndef WOLFRAM_DIALECT
#define WOLFRAM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Wolfram dialect definition.
//===----------------------------------------------------------------------===//

def Wolfram_Dialect : Dialect {
  let name = "wolfram"; // Name of the dialect
  let summary = "A dialect for the Wolfram Language"; // Short description
  let description = [{
    The Wolfram dialect represents constructs from the Wolfram Language,
    such as expressions, functions, and symbolic computation.
  }];
  let cppNamespace = "::mlir::wolfram"; // C++ namespace for the dialect

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base wolfram operation definition.
//===----------------------------------------------------------------------===//

class WolframDialectOp<string mnemonic, list<Trait> traits = []> :
        Op<Wolfram_Dialect, mnemonic, traits>;

#endif // WOLFRAM_DIALECT
