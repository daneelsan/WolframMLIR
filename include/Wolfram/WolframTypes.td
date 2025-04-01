#ifndef WOLFRAM_TYPES
#define WOLFRAM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Wolfram/WolframDialect.td"

// Base class for Wolfram types
class Wolfram_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Wolfram_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Define the Wolfram integer type
def Wolfram_IntegerType : Wolfram_Type<"Integer", "integer"> {
  let summary = "A Wolfram integer type";
  let description = [{
    Represents a Wolfram integer.
  }];
}

// Define the Wolfram real type
def Wolfram_RealType : Wolfram_Type<"Real", "real"> {
  let summary = "A Wolfram real type";
  let description = [{
    Represents a Wolfram real number.
  }];
}

// Define the Wolfram string type
def Wolfram_StringType : Wolfram_Type<"String", "string"> {
  let summary = "A Wolfram string type";
  let description = [{
    Represents a Wolfram string.
  }];
}

// Define the Wolfram boolean type
def Wolfram_BooleanType : Wolfram_Type<"Boolean", "boolean"> {
  let summary = "A Wolfram boolean type";
  let description = [{
    Represents a Wolfram boolean.
  }];
}

// Define the Wolfram symbol type
def Wolfram_SymbolType : Wolfram_Type<"Symbol", "symbol"> {
  let summary = "A Wolfram symbol type";
  let description = [{
    Represents a Wolfram symbol.
  }];
}

// Define the Wolfram function type
def Wolfram_FunctionType : Wolfram_Type<"Function", "function"> {
  let summary = "A Wolfram function type";
  let description = [{
    Represents a Wolfram function.
  }];
}

// Define the Wolfram expression type
def Wolfram_ExprType : Wolfram_Type<"Expr", "expr"> {
  let summary = "A Wolfram expression type";
  let description = [{
    Represents a generic Wolfram expression.
  }];
}

#endif // WOLFRAM_TYPES