#ifndef WOLFRAM_C_DIALECTS_H
#define WOLFRAM_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Wolfram, wolfram);

#ifdef __cplusplus
}
#endif

#endif // WOLFRAM_C_DIALECTS_H
