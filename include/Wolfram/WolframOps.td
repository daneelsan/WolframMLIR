
#ifndef WOLFRAMOPS_TD
#define WOLFRAMOPS_TD

include "Wolfram/WolframOpBase.td"
include "Wolfram/WolframTypes.td"

include "mlir/Bytecode/BytecodeOpInterface.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "mlir/IR/SymbolInterfaces.td"

// include "mlir/Dialect/LLVMIR/LLVMDialect.td"
// include "mlir/Dialect/LLVMIR/LLVMOps.td"

class Wolfram_Builder<string builder> {
  string wolframBuilder = builder;
}

def Wolfram_ConstantOp : Wolfram_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Defines a constant of Wolfram type";
  
  let arguments = (ins SI64Attr:$value);
  let results = (outs WolframInteger64Type:$res);
  
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
}

def Wolfram_LoadArgumentOp : Wolfram_Op<"load_argument", [Pure]> {
  let arguments = (ins StrAttr:$arg_name);
  let results = (outs WolframInteger64Type:$result);
  
  let assemblyFormat = "$arg_name attr-dict `:` type($result)";
}

def Wolfram_PlusOp : Wolfram_Op<"plus", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins WolframInteger64Type:$lhs, WolframInteger64Type:$rhs);
  let results = (outs WolframInteger64Type:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Wolfram_ReturnOp : Wolfram_Op<"return", [Terminator]> {
  let arguments = (ins WolframInteger64Type:$value);
  
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

// def Wolfram_FuncOp : Wolfram_Op<"func", [
//     IsolatedFromAbove,
//     SingleBlockImplicitTerminator<"ReturnOp">,
//     Symbol
// ]> {
//   let summary = "Wolfram function operation";
//   let description = [{
//     Defines a Wolfram function with a body and return type.
//   }];

//   let arguments = (ins
//     StrAttr:$sym_name,
//     TypeAttr:$function_type
//   );
//   let regions = (region AnyRegion:$body);

//   // let assemblyFormat = [{
//   //   $sym_name `(` `)` `:` `(` `)` `->` `(` type($function_type) `)` attr-dict $body
//   // }];

//   let builders = [
//     OpBuilder<(ins "StringRef":$name, "FunctionType":$type)>
//   ];

//   let extraClassDeclaration = [{
//     // Add entry block with arguments matching the function type
//     Block *addEntryBlock();
    
//     // Get the function type
//     FunctionType getFunctionType() {
//       return cast<FunctionType>(getFunctionTypeAttr().getValue());
//     }
//   }];
// }

// Base class for Wolfram terminator operations.
// All terminator operations have zero results and an optional list of successors.
// class Wolfram_TerminatorOp<string mnemonic, list<Trait> traits = []> :
//     Wolfram_Op<mnemonic, !listconcat(traits, [Terminator])>;


// def Wolfram_ConstantOp
//     : Wolfram_Op<"mlir.constant", [Pure, ConstantLike]>,
//       Wolfram_Builder<[{$res = getWolframConstant($_resultType, $value, $_location, moduleTranslation);}]>
// {
//   let summary = "Defines a constant Wolfram type";

//   let arguments = (ins AnyAttr:$value);
//   let results = (outs LLVM_Type:$res);

//   let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";

  // let builders = [
  //   //LLVM_OneResultOpBuilder,
  //   // OpBuilder<(ins "Type":$type, "int64_t":$value), [{
  //   //   build($_builder, $_state, type, $_builder.getIntegerAttr(type, value));
  //   // }]>
  // ];

  // let extraClassDeclaration = [{
  //   /// Whether the constant op can be constructed with a particular value and
  //   /// type.
  //   static bool isBuildableWith(Attribute value, Type type);

  //   /// Build the constant op with `value` and `type` if possible, otherwise
  //   /// returns null.
  //   static ConstantOp materialize(OpBuilder &builder, Attribute value,
  //                                 Type type, Location loc);
  // }];

  // let hasFolder = 1;
  // let hasVerifier = 1;
// }

// def Wolfram_LoadArgumentOp : WolframDialectOp<"load_argument"> {
//   let arguments = (ins StrAttr:$name);
//   let results = (outs Wolfram_AnyType:$result);
//   let assemblyFormat = "`(` $name `)` attr-dict `:` type($result)";
// }

// def Wolfram_CallOp : WolframDialectOp<"call"> {
//   let arguments = (ins StrAttr:$callee, Variadic<Wolfram_AnyType>:$args);
//   let results = (outs Wolfram_AnyType:$result);
//   let assemblyFormat = "$callee `(` $args `)` attr-dict `:` type($args) `->` type($result)";
// }

// // Unconditional branch
// def Wolfram_BrOp : Wolfram_TerminatorOp<"br", 
//     [DeclareOpInterfaceMethods<BranchOpInterface>, Pure]> {
//   let arguments = (ins
//     Variadic<Wolfram_AnyType>:$destOperands
//   );
//   let successors = (successor AnySuccessor:$dest);
//   let assemblyFormat = [{
//     $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
//   }];
//   let builders = [
//     OpBuilder<(ins "Block *":$dest), [{
//       build($_builder, $_state, ValueRange(), dest);
//     }]>,
//     OpBuilder<(ins "ValueRange":$operands, "Block *":$dest), [{
//       build($_builder, $_state, operands, dest);
//     }]>
//   ];
// }

// // Conditional branch
// def Wolfram_CondBrOp : Wolfram_TerminatorOp<"cond_br",
//     [AttrSizedOperandSegments,
//      DeclareOpInterfaceMethods<BranchOpInterface>,
//      Pure]> {
//   let arguments = (ins 
//     Wolfram_BoolType:$condition,
//     Variadic<Wolfram_AnyType>:$trueDestOperands,
//     Variadic<Wolfram_AnyType>:$falseDestOperands
//   );
//   let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
//   let assemblyFormat = [{
//     $condition `,` 
//     $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
//     $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
//     attr-dict
//   }];
//   let builders = [
//     OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
//       "ValueRange":$trueOperands, "Block *":$falseDest,
//       "ValueRange":$falseOperands), [{
//       build($_builder, $_state, condition, trueDest, trueOperands,
//             falseDest, falseOperands);
//     }]>
//   ];
// }

// def Wolfram_ReturnOp : Wolfram_TerminatorOp<"return", [Pure, ReturnLike]> {
//   let arguments = (ins Optional<Wolfram_Type>:$arg);
//   let assemblyFormat = "attr-dict ($arg^ `:` type($arg))?";
  
//   let builders = [
//     OpBuilder<(ins), [{
//       build($_builder, $_state, ValueRange());
//     }]>,
//     OpBuilder<(ins "Value":$arg), [{
//       build($_builder, $_state, arg);
//     }]>
//   ];
  
//   let hasVerifier = 1;
// }

#endif // WOLFRAMOPS_TD