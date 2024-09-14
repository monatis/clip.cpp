cmake_minimum_required(VERSION 3.22.1)

project("clip-android")

# Add local clip.cpp CMake project
add_subdirectory(../../../../../../ build-clip)

add_library(
        ${CMAKE_PROJECT_NAME}
        SHARED
        clip_android.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        clip
        ggml
        android
        log)