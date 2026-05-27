cmake_minimum_required(VERSION 3.28)
project(mlask)

include(FetchContent)
FetchContent_Declare(Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(Eigen)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG        v21.12
)
FetchContent_MakeAvailable(protobuf)

set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ONNX_USE_LITE_PROTO OFF CACHE BOOL "" FORCE)

set(ONNX_CUSTOM_PROTOC_EXECUTABLE $<TARGET_FILE:protoc> CACHE STRING "" FORCE)

set(IR_VERSION 8)
set(OPSET_VERSION 14)
configure_file(config.h.in config/config.h)

FetchContent_Declare(
    onnx
    GIT_REPOSITORY https://github.com/onnx/onnx.git
    GIT_TAG        v1.14.1
)
FetchContent_MakeAvailable(onnx)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(mlask STATIC
    src/Model.cpp
    src/Relu.cpp
    src/ActivationFunction.cpp
)
target_compile_features(mlask PRIVATE cxx_std_23)
target_link_libraries(mlask PUBLIC Eigen3::Eigen
    onnx
    onnx_proto
    protobuf::libprotobuf
)
include_directories(${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/config
    ${onnx_SOURCE_DIR}
    ${onnx_BINARY_DIR}
)

add_subdirectory(tests)
