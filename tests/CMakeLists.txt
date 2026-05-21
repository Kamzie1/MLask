add_executable(Layer Layer.cpp)
target_compile_features(Layer PRIVATE cxx_std_23)
target_link_libraries(Layer PRIVATE mlask)
include_directories(${CMAKE_SOURCE_DIR}/include)
