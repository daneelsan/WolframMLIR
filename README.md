

## File Structure

├── CMakeLists.txt                     # Top-level CMake configuration
├── README.md                          # Project Documentation
├── include
│   ├── CMakeLists.txt                 # CMake for the include directory
│   ├── Wolfram
│   │   ├── CMakeLists.txt             # CMake for the Wolfram dialect headers
│   │   ├── WolframDialect.h           # Handwritten dialect header
│   │   ├── WolframDialect.td          # TableGen dialect definition
│   │   ├── WolframOps.h               # Handwritten operations header
│   │   ├── WolframOps.td              # TableGen operations definition
│   │   ├── WolframTypes.h             # Handwritten types header
│   │   └── WolframTypes.td            # TableGen types definition
│   └── Wolfram-c
│       └── Dialects.h                 # C API header
├── lib
│   ├── CAPI
│   │   └── Dialects.cpp               # C API implementation
│   ├── CMakeLists.txt                 # CMake for the lib directory
│   └── Wolfram
│       ├── CMakeLists.txt             # CMake for the Wolfram dialect implementation
│       ├── WolframDialect.cpp         # Dialect implementation
│       ├── WolframOps.cpp             # Operations implementation
│       └── WolframTypes.cpp           # Types implementation

## Build

### Build MLIR

```shell
$ echo $LLVM_SRC_DIR                                                                                                                              
/Users/daniels/git/llvm-project
```

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

### Build Wolfram MLIR

```shell
$ rm -rf build

$ cmake -S . -B build -DMLIR_DIR=$LLVM_SRC_DIR/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_SRC_DIR/build/bin/llvm-lit

$ cmake --build build
```