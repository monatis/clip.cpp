add_library(common-clip STATIC common-clip.h common-clip.cpp)
target_include_directories(common-clip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(common-clip PUBLIC cxx_std_11)
target_link_libraries(common-clip PRIVATE ggml clip)

if (CLIP_BUILD_IMAGE_SEARCH)
    add_subdirectory(./image-search)
endif()

add_executable(main main.cpp)
target_compile_features(main PUBLIC cxx_std_11)
target_link_libraries(main PRIVATE clip common-clip ggml)

add_executable(zsl zsl.cpp)
target_compile_features(zsl PUBLIC cxx_std_11)
target_link_libraries(zsl PRIVATE clip common-clip ggml)

add_executable(simple_c simple.c)
target_compile_features(simple_c PUBLIC c_std_11)
target_link_libraries(simple_c PRIVATE clip ggml)

add_executable(extract extract.cpp)
target_compile_features(extract PUBLIC cxx_std_11)
target_link_libraries(extract PRIVATE clip common-clip ggml)
