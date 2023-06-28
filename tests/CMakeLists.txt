add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE clip common-clip ggml)
