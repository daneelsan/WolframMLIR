#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

#include "Wolfram/WolframDialect.h"
#include "Wolfram/WolframPasses.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

/// Dialect plugin registration mechanism.
/// Observe that it also allows to register passes.
/// Necessary symbol to register the dialect plugin.
extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo()
{
  llvm::errs() << "[TEST] Loading Wolfram dialect plugin...\n";
  return {MLIR_PLUGIN_API_VERSION, "Wolfram", LLVM_VERSION_STRING,
          [](DialectRegistry *registry)
          {
            registry->insert<mlir::wolfram::WolframDialect>();
            mlir::wolfram::registerPasses();
          }};
}

/// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo()
{
  return {MLIR_PLUGIN_API_VERSION, "WolframPasses", LLVM_VERSION_STRING,
          []()
          { mlir::wolfram::registerPasses(); }};
}
