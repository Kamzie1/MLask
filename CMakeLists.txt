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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(mlask STATIC 
    src/Model.cpp
)
target_compile_features(mlask PRIVATE cxx_std_23)
target_link_libraries(mlask PUBLIC Eigen3::Eigen)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(tests)
