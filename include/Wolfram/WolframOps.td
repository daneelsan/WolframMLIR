
#ifndef WOLFRAM_OPS
#define WOLFRAM_OPS

include "Wolfram/WolframTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Bytecode/BytecodeOpInterface.td"

// Define the AddOp (for +)
def Wolfram_AddOp : WolframDialectOp<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Add two Wolfram numbers";
  let description = [{
    Adds two Wolfram numbers (integers or reals).
  }];

  let arguments = (ins Wolfram_IntegerType:$lhs, Wolfram_IntegerType:$rhs);
  let results = (outs Wolfram_IntegerType:$result);

  let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` type($result)";
}

// Define the FunctionOp
def Wolfram_FunctionOp : WolframDialectOp<"function"> {
  let summary = "Define a Wolfram function";
  let description = [{
    Defines a Wolfram function with a list of parameters and a body.
  }];

  let arguments = (ins StrArrayAttr:$params, Wolfram_ExprType:$body);
  let results = (outs Wolfram_FunctionType:$result);

  let assemblyFormat = "`(` $params `)` `->` $body attr-dict `:` type($result)";
}

def Wolfram_FuncOp : WolframDialectOp<"func", [IsolatedFromAbove]> {
  let summary = "Define a Wolfram function";
  let description = [{
    Defines a Wolfram function with a list of parameters and a body.
  }];

  let arguments = (ins Variadic<Wolfram_SymbolType>:$params, Wolfram_ExprType:$body);
  let results = (outs Wolfram_FunctionType:$result);

  //let assemblyFormat = "`(` $params `)` `->` $body attr-dict `:` type($result)";
}

// Define the ApplyOp
def Wolfram_ApplyOp : WolframDialectOp<"apply"> {
  let summary = "Apply a Wolfram function to arguments";
  let description = [{
    Applies a Wolfram function to a list of arguments.
  }];

  let arguments = (ins Wolfram_FunctionType:$func, Variadic<Wolfram_ExprType>:$args);
  let results = (outs Wolfram_ExprType:$result);

  let assemblyFormat = "`(` $func `,` $args `)` attr-dict `:` type($result)";
}

// Define the CallOp
def Wolfram_CallOp : WolframDialectOp<"call", []> {
  let summary = "Call a Wolfram function";
  let description = [{
    Calls a Wolfram function with a list of arguments.
  }];

  let arguments = (ins Wolfram_FunctionType:$callee, Variadic<Wolfram_ExprType>:$args);
  let results = (outs Wolfram_ExprType:$result);

  let assemblyFormat = "$callee `(` $args `)` attr-dict `:` type($result)";
}

// Define the ConstantOp for integers
def Wolfram_ConstantIntOp : WolframDialectOp<"constant.int"> {
  let summary = "Define a constant integer";
  let description = [{
    Represents a constant integer value.
  }];

  let arguments = (ins I64Attr:$value);
  let results = (outs Wolfram_IntegerType:$result);

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
}

// Define the ConstantOp for generic expressions
def Wolfram_ConstantOp : WolframDialectOp<"constant", [ConstantLike]> {
  let summary = "Define a constant value";
  let description = [{
    Represents a constant value of a specific type.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs Wolfram_ExprType:$result);

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
}

// Define the SymbolOp
def Wolfram_SymbolOp : WolframDialectOp<"symbol"> {
  let summary = "Define a Wolfram symbol";
  let description = [{
    Represents a Wolfram symbol, such as a variable or function name.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs Wolfram_SymbolType:$result);

  let assemblyFormat = "`(` $name `)` attr-dict `:` type($result)";
}

#endif // WOLFRAM_OPS