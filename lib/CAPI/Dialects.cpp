#include "Wolfram-c/Dialects.h"

#include "Wolfram/WolframDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Wolfram, wolfram,
                                      mlir::wolfram::WolframDialect)
