# Wolfram MLIR Dialect

This project implements an MLIR dialect for Wolfram Language IR, with a path to lower to LLVM IR.
The dialect closely resembles LLVM IR to enable efficient compilation of Wolfram Language code.


## Build

### Build MLIR

`$LLVM_SRC_DIR` stores the directory of the `llvm-project` repository:
```shell
$ echo $LLVM_SRC_DIR
/Users/daniels/git/llvm-project
```

Clone the LLVM git project (I chose to put it in `~/git`):
```shell
$ cd ~/git && git clone https://github.com/llvm/llvm-project.git && cd llvm-project
```

Create a directory to store the MLIR build:
```
$ mkdir build-mlir && cd build-mlir
```

Configure the build using `cmake` (see https://mlir.llvm.org/getting_started/):
```shell
$ cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_LLD=ON -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD=host \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON

$ cmake --build . --target check-mlir
```

`$MLIR_BUILD_DIR` stores the directory where MLIR was build:
```shell
$ echo $MLIR_BUILD_DIR
/Users/daniels/git/llvm-project/build-mlir
```

### Build the Wolfram Dialect (Out-of-Tree)

Now that MLIR is built we proceed to build this project:
```shell
$ rm -rf build

$ cmake -S . -B build -DMLIR_DIR=$MLIR_BUILD_DIR/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$MLIR_BUILD_DIR/bin/llvm-lit

$ cmake --build build
```

## Project Structure

```
├── CMakeLists.txt                     # Top-level CMake configuration
├── README.md                          # Project Documentation
├── include                            # Public headers
│   ├── CMakeLists.txt
│   ├── Wolfram
│   │   ├── CMakeLists.txt
│   │   ├── WolframDialect.h           # Main dialect class
│   │   ├── WolframDialect.td
│   │   ├── WolframOpBase.td
│   │   ├── WolframOps.h               # Generated ops
│   │   ├── WolframOps.td
│   │   ├── WolframPasses.h            # Pass declarations
│   │   ├── WolframPasses.td
│   │   ├── WolframTypes.h             # Type system
│   │   └── WolframTypes.td
│   └── Wolfram-c
│       └── Dialects.h
├── lib
│   ├── CAPI
│   │   └── Dialects.cpp
│   ├── CMakeLists.txt
│   └── Wolfram                        # Dialect implementation
│       ├── CMakeLists.txt
│       ├── WolframDialect.cpp         # Dialect registration
│       ├── WolframOps.cpp             # Op implementations
│       ├── WolframPasses.cpp          # Pass implementations
│       └── WolframTypes.cpp           # Type implementations
├── test                               # Test cases
│   ├── CMakeLists.txt
│   ├── Wolfram
│   │   ├── dummy.mlir
│   │   ├── wolfram-opt.mlir
│   │   └── wolfram-plugin.mlir
│   ├── lit.cfg.py
│   └── lit.site.cfg.py.in
├── wolfram-opt                        # Optimization tool
│   ├── CMakeLists.txt
│   └── wolfram-opt.cpp
└── wolfram-plugin                     # Plugin for mlir-opt
    ├── CMakeLists.txt
    └── wolfram-plugin.cpp
```

### TableGen Overview

[TableGen](https://llvm.org/docs/TableGen/) is LLVM's DSL for generating:
- Operation definitions
- Type definitions
- Pass declarations
- Documentation

Key concepts:
- `def`: Defines a new record (operation, type, etc.)
- `let`: Sets properties of a record
- `class`: Abstract definition that can be inherited

### Key Files

1. **Header Files (`include/Wolfram/*.h`)**
    - `WolframDialect.h`: Main dialect class declaration
    - `WolframOps.h`: Auto-generated operation class declarations
    - `WolframTypes.h`: Type system interface
    - `WolframPasses.h`: Pass management declarations

2. **[TableGen](https://llvm.org/docs/TableGen/) Definition Files (`include/Wolfram/*.td`)**
    - `WolframDialect.td`: Defines the dialect's properties (name, summary, C++ namespace)
    - `WolframOps.td`: Declares all Wolfram operations (like `constant`, `plus`, `return`) with their:
        - Input/output types
        - Assembly format
        - Traits (e.g., `Pure`, `Terminator`)
    - `WolframTypes.td`: Defines Wolfram-specific types (currently just `!wolfram.i64`)
    - `WolframPasses.td`: Declares transformation passes (like `wolfram-switch-bar-foo`)


3. **Implementation Files (`lib/Wolfram/*.cpp`)**
    - `WolframDialect.cpp`: Registers the dialect and its components
    - `WolframOps.cpp`: Operation implementations (currently minimal)
    - `WolframTypes.cpp`: Type registration and utilities
    - `WolframPasses.cpp`: Pass implementations (like the bar→foo rewriter)
    - `WolframToLLVM.cpp`: Conversion patterns to LLVM dialect (*TODO*)
    - `WolframLowering.cpp`: Lowering infrastructure (*TODO*)

4. **CAPI Files**
    - `Dialects.cpp`: C-API registration for language bindings

### Key Tools

#### `wolfram-opt`

**Purpose**: Main development tool for:

- Testing dialect functionality
- Running optimization passes
- Performing dialect conversions

**Key Capabilities**:

```
# Show registered dialects
$ ./build/bin/wolfram-opt --show-dialects                                                                                                         
Available Dialects: arith,builtin,func,wolfram

# Run specific passes (not working)
$ ./build/bin/wolfram-opt test/Wolfram/dummy.mlir --wolfram-switch-bar-foo

# Verify conversion pipeline (TODO)
$ ./build/bin/wolfram-opt test/Wolfram/dummy.mlir --wolfram-to-llvm
```

#### `WolframPlugin`

**Purpose**: Allows using the Wolfram dialect with standard MLIR tools like `mlir-opt`

**Key Features**:

- Dynamically loads the dialect into existing tools
- Enables testing without recompiling main tools
- Usage:
    ```shell
    $ $MLIR_BUILD_DIR/bin/mlir-opt --load-dialect-plugin=build/lib/WolframPlugin.dylib test/Wolfram/dummy.mlir
    [TEST] Loading Wolfram dialect plugin...
    [TEST] Registering Wolfram dialect operations...
    module {
    %0 = wolfram.constant(1) : !wolfram.i64
    %1 = wolfram.constant(2) : !wolfram.i64
    %2 = wolfram.plus %1, %0 : !wolfram.i64
    wolfram.return %2 : !wolfram.i64
    }
    ```

## Typical Development Flow

1. **Add New Operation**:
    - Define in `WolframOps.td`
    - Rebuild with `ninja WolframOpsIncGen`
    - Test via `wolfram-opt` or plugin

2. **Add Conversion**:
    - Implement in `WolframToLLVM.cpp`
    - Register in conversion pass
    - Test with `--wolfram-to-llvm`

3. **Add Optimization**:
    - Declare pass in `WolframPasses.td`
    - Implement in `WolframPasses.cpp`
    - Test with `wolfram-opt --my-pass`

## Conversion Pipeline (TODO)

To achieve Wolfram → LLVM lowering:

1. **Implement Conversion Patterns:**
    - Create `WolframToLLVM.cpp` with patterns for:
    - `wolfram.constant` → `llvm.mlir.constant`
    - `wolfram.plus` → `llvm.add`
    - `wolfram.return` → `llvm.return`

2. **Set Up Lowering Pipeline:**
    ```
    pm.addPass(createWolframToLLVMLoweringPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    ```

3. **Type Conversion:**
    - Implement type converter for `!wolfram.i64` → `i64` (or use basic MLIR types directly)

4. **Test Infraestructure**
    - Add lit tests for conversion
    - Verify with `mlir-translate --mlir-to-llvmir`

## Current Status

- [X] Basic dialect infrastructure
- [ ] Operation definitions
- [ ] Type system
- [ ] Pass infrastructure
- [ ] Wolfram → LLVM conversion
- [ ] Full lowering pipeline
- [ ] Optimization passes

## References

- https://mlir.llvm.org/docs/DefiningDialects/
    - https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/
    - https://mlir.llvm.org/docs/DefiningDialects/Operations/
- https://mlir.llvm.org/docs/Tutorials/CreatingADialect/
- https://mlir.llvm.org/docs/LangRef/
- https://llvm.org/devmtg/2020-09/slides/MLIR_Tutorial.pdf
- https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone
- https://www.pure.ed.ac.uk/ws/portalfiles/portal/157150307/A_functional_LUCKE_DOA20052020_AFV.pdf
- [How to Build your own MLIR Dialect](https://av.tib.eu/media/61396)