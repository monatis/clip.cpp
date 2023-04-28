add_executable(quantize quantize.cpp common-ggml.h common-ggml.cpp)
target_link_libraries(quantize PRIVATE clip ggml)