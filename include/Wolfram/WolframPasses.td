#ifndef WOLFRAM_PASS
#define WOLFRAM_PASS

include "mlir/Pass/PassBase.td"

def WolframSwitchBarFoo: Pass<"wolfram-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

#endif // WOLFRAM_PASS
