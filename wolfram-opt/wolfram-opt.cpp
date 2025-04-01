#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Wolfram/WolframDialect.h"
#include "Wolfram/WolframPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::wolfram::registerPasses();
  // TODO: Register wolfram passes here.

  mlir::DialectRegistry registry;
  registry.insert<mlir::wolfram::WolframDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Wolfram optimizer driver\n", registry));
}
