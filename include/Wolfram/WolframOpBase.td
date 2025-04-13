#ifndef WOLFRAMOPBASE_TD
#define WOLFRAMOPBASE_TD

include "Wolfram/WolframDialect.td"

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Wolfram dialect type constraints.
//===----------------------------------------------------------------------===//

// Wolfram dialect type.
// def Wolfram_Type : DialectType<Wolfram_Dialect,
//                             CPred<"::mlir::wolfram::isCompatibleOuterType($_self)">,
//                             "Wolfram dialect-compatible type">;

// Base class for Wolfram operations.
// Defines the interface to the llvm::IRBuilder used to translate to proper Wolfram 
// IR and the interface to the mlir::OpBuilder used to import from Wolfram IR.
class Wolfram_OpBase<Dialect dialect, string mnemonic, list<Trait> traits = []> :
    Op<dialect, mnemonic, traits> {
  // A pattern for constructing the LLVM IR Instruction (or other Value) that
  // corresponds to this op.  This pattern can use `builder` to refer to an
  // `llvm::IRBuilder<>` instance, $-names of arguments and results and the
  // following special variable names:
  //   - $_resultType - substituted with the LLVM IR type of the result;
  //   - $_numOperands - substituted with the number of operands (including
  //                     the variadic ones);
  //   - $_hasResult - substituted with a check that a variadic-result op does
  //                   have a result (LLVM ops can have 0 or 1 result);
  //   - $_location - mlir::Location object of the instruction.
  // Additionally, `$$` can be used to produce the dollar character.
  string wolframBuilder = "";

  // A builder to construct the MLIR LLVM dialect operation given the matching
  // LLVM IR instruction `inst` and its operands `llvmOperands`. The
  // following $-variables exist:
  //   - $name - substituted by the remapped `inst` operand value at the index
  //             of the MLIR operation argument with the given name, or if the
  //             name matches the result name, by a reference to store the
  //             result of the newly created MLIR operation to;
  //   - $_op - substituted by a reference to store the newly created MLIR
  //            operation (only for MLIR operations that return no result);
  //   - $_int_attr - substituted by a call to an integer attribute matcher;
  //   - $_float_attr - substituted by a call to a float attribute matcher;
  //   - $_var_attr - substituted by a call to a variable attribute matcher;
  //   - $_label_attr - substituted by a call to a label attribute matcher;
  //   - $_roundingMode_attr - substituted by a call to a rounding mode
  //     attribute matcher;
  //   - $_fpExceptionBehavior_attr - substituted by a call to a FP exception
  //     behavior attribute matcher;
  //   - $_resultType - substituted with the MLIR result type;
  //   - $_location - substituted with the MLIR location;
  //   - $_builder - substituted with the MLIR builder;
  //   - $_qualCppClassName - substitiuted with the MLIR operation class name.
  // Always either store a reference to the result of the newly created
  // operation, or to the operation itself if it does not return a result.
  // Additionally, `$$` can be used to produce the dollar character.
  string mlirBuilder = "";

  // An array that specifies a mapping from MLIR argument indices to Wolfram IR
  // operand indices. The mapping is necessary since argument and operand
  // indices do not always match. If not defined, the array is set to the
  // identity permutation. An operation may define any custom index permutation
  // and set a specific argument index to -1 if it does not map to an LLVM IR
  // operand.
  list<int> wolframArgIndices = [];
}

//===----------------------------------------------------------------------===//
// Base classes for Wolfram dialect operations.
//===----------------------------------------------------------------------===//

// Base class for LLVM operations. All operations get an "llvm." prefix in
// their name automatically and should either have zero or one result.
class Wolfram_Op<string mnemonic, list<Trait> traits = []> :
    Wolfram_OpBase<Wolfram_Dialect, mnemonic, traits>;

#endif  // WOLFRAMOPBASE_TD
