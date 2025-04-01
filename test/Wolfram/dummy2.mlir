module {
  // Define the function f = Function[{u, v}, u + v + 42]
  func.func @f(%u: !wolfram.integer, %v: !wolfram.integer) -> !wolfram.integer {
    // Define constants
    %42 = wolfram.constant.int (42) : !wolfram.integer

    // Perform addition: u + v
    %sum1 = wolfram.add (%u, %v) : !wolfram.integer

    // Perform addition: (u + v) + 42
    %sum2 = wolfram.add (%sum1, %42) : !wolfram.integer

    // Return the result
    func.return %sum2 : !wolfram.integer
  }

  // Apply the function f[1, 2]
  %1 = wolfram.constant.int (1) : !wolfram.integer
  %2 = wolfram.constant.int (2) : !wolfram.integer
  %result = func.call @f(%1, %2) : (!wolfram.integer, !wolfram.integer) -> !wolfram.integer

  // Print the result (optional, if you have a print operation)
  // wolfram.print %result : !wolfram.integer
}