#ifndef WOLFRAMTYPES_TD
#define WOLFRAMTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Wolfram/WolframDialect.td"

// Base class for Wolfram dialect types
class WolframType<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Wolfram_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

def WolframInteger64Type : WolframType<"Integer64", "i64"> {
  let summary = "Wolfram 64-bit integer type";
}

def WolframReal64Type : WolframType<"Real64", "f64"> {
  let summary = "Wolfram 64-bit floating-point type";
}

// def WolframBooleanType : WolframType<"Boolean", "bool"> {
//   let summary = "Wolfram Boolean type";
// }

// def WolframFunctionType : WolframType<"FuncType", "func"> {
//   let summary = "Function type";
//   let parameters = (ins "Type":$input, "Type":$output);
// }

#endif // WOLFRAMTYPES_TD
