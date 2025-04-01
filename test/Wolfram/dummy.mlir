// RUN: mlir-opt %s --load-dialect-plugin=%wolfram_libs/WolframPlugin%shlibext | FileCheck %s

module {
  // Define the function: Function[{u, v}, u + v + 42]
  // wolfram.func @f(%u: !wolfram.integer, %v: !wolfram.integer) -> !wolfram.integer {
  //   %sum1 = wolfram.add %u, %v : !wolfram.integer
  //   %forty_two = wolfram.constant.int 42 : !wolfram.integer
  //   %result = wolfram.add %sum1, %forty_two : !wolfram.integer
  //   wolfram.return %result : !wolfram.integer
  // }

  // Apply the function `f` to arguments `1` and `2`
  %one = wolfram.constant.int (1) : !wolfram.integer
  %two = wolfram.constant.int (2) : !wolfram.integer
  %result = wolfram.add (%one, %two) : !wolfram.integer
  //%result = wolfram.apply @f, %one, %two : (!wolfram.function, !wolfram.integer, !wolfram.integer) -> !wolfram.integer

  // Return the result
  wolfram.return: !wolfram.integer
}