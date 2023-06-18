add_executable(main main.cpp)
target_link_libraries(main PRIVATE clip ggml)

add_executable(zsl zsl.cpp)
target_link_libraries(zsl PRIVATE clip ggml)
