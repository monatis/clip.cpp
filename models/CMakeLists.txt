add_executable(quantize quantize.cpp)
target_link_libraries(quantize PRIVATE clip ggml)
