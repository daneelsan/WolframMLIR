#ifndef WOLFRAM_WOLFRAMPASSES_H
#define WOLFRAM_WOLFRAMPASSES_H

#include "Wolfram/WolframDialect.h"
#include "Wolfram/WolframOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir
{
    namespace wolfram
    {
#define GEN_PASS_DECL
#include "Wolfram/WolframPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Wolfram/WolframPasses.h.inc"
    } // namespace wolfram
} // namespace mlir

#endif
