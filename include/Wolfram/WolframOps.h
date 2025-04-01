#ifndef WOLFRAM_WOLFRAMOPS_H
#define WOLFRAM_WOLFRAMOPS_H

#include "Wolfram/WolframTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#define GET_OP_CLASSES
#include "Wolfram/WolframOps.h.inc"

#endif // WOLFRAM_WOLFRAMOPS_H
