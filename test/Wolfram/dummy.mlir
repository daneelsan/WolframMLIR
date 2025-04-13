module {
  %one = wolfram.constant(1) : !wolfram.i64
  %two = wolfram.constant(2) : !wolfram.i64
  %result = wolfram.plus %two, %one : !wolfram.i64
  wolfram.return %result : !wolfram.i64
}