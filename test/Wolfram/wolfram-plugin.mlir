// RUN: mlir-opt %s --load-dialect-plugin=%wolfram_libs/WolframPlugin%shlibext --pass-pipeline="builtin.module(wolfram-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @wolfram_types(%arg0: !wolfram.custom<"10">)
  func.func @wolfram_types(%arg0: !wolfram.custom<"10">) {
    return
  }
}
