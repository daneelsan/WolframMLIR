#include "Wolfram/WolframTypes.h"
#include "Wolfram/WolframDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::wolfram;

#define GET_TYPEDEF_CLASSES
#include "Wolfram/WolframOpsTypes.cpp.inc"

void WolframDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "Wolfram/WolframOpsTypes.cpp.inc"
        >();
}
