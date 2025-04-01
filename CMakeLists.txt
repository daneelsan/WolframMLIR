# Specify the minimum required version of CMake
cmake_minimum_required(VERSION 3.20.0)

# Define the project name and the languages used (C++ and C)
project(wolfram-dialect LANGUAGES CXX C)

# Ensure that the build uses the install name directory
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

# Set the C++ standard to C++17 and cache the value
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# Check if the current source directory is the root source directory
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Find the MLIR package, which is required for this project
  find_package(MLIR REQUIRED CONFIG)

  # Print the paths to MLIRConfig.cmake and LLVMConfig.cmake for debugging
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  # Set output directories for LLVM runtime and libraries
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  # Add MLIR and LLVM CMake module paths to the CMAKE_MODULE_PATH
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  # Include necessary CMake modules for TableGen, LLVM, MLIR, and LLVM options
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # If building via external projects mechanism, set MLIR source and include directories
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

# Set source and binary directories for the wolfram-mlir project
set(WOLFRAM_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(WOLFRAM_BINARY_DIR ${PROJECT_BINARY_DIR})

# Include necessary directories for headers and libraries
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${WOLFRAM_SOURCE_DIR}/include)
include_directories(${WOLFRAM_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})

# Add LLVM definitions to the project
add_definitions(${LLVM_DEFINITIONS})

# Add subdirectories for the project components
add_subdirectory(include)  # Include directory for headers
add_subdirectory(lib)      # Library directory for source files

# Add subdirectories for tests, wolfram-opt, wolfram-plugin, and wolfram-translate
add_subdirectory(test)
add_subdirectory(wolfram-opt)
add_subdirectory(wolfram-plugin)
#add_subdirectory(wolfram-translate)
