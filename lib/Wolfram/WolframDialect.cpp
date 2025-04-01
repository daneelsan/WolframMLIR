#include "Wolfram/WolframDialect.h"
#include "Wolfram/WolframOps.h"
#include "Wolfram/WolframTypes.h"

using namespace mlir;
using namespace mlir::wolfram;

#include "Wolfram/WolframOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Wolfram dialect.
//===----------------------------------------------------------------------===//

void WolframDialect::initialize()
{
    llvm::errs() << "[TEST] Registering Wolfram dialect operations...\n";
    addOperations<
#define GET_OP_LIST
#include "Wolfram/WolframOps.cpp.inc"
        >();
    registerTypes();
}
