// build/bin/mlir-tblgen -gen-dialect-decls ~/projects/thesis/wolfram-mlir/tablegen/WolframDialect.td ~/projects/thesis/wolfram-mlir/include/Wolfram/Ops.td -I mlir/include

#ifndef WOLFRAMDIALECT_TD
#define WOLFRAMDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Wolfram dialect definition.
//===----------------------------------------------------------------------===//

def Wolfram_Dialect : Dialect {
  let name = "wolfram";
  let summary = "A dialect for the Wolfram Language IR";
  let description = [{
    A dialect representing the Wolfram Language's intermediate representation.
  }];
  let cppNamespace = "::mlir::wolfram"; // C++ namespace for the dialect

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // WOLFRAMDIALECT_TD
