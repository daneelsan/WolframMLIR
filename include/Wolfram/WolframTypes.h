#ifndef WOLFRAM_WOLFRAMTYPES_H
#define WOLFRAM_WOLFRAMTYPES_H

#include "mlir/IR/BuiltinTypes.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Wolfram/WolframOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

// namespace mlir
// {
//     namespace wolfram
//     {

//         /// Returns `true` if the given type is compatible with the Wolfram dialect. This
//         /// is an alias to `WolframDialect::isCompatibleType`.
//         bool isCompatibleType(Type type);

//         /// Returns `true` if the given outer type is compatible with the Wolfram dialect
//         /// without checking its potential nested types such as struct elements.
//         bool isCompatibleOuterType(Type type);

//     }
// }

#endif // WOLFRAM_WOLFRAMTYPES_H
