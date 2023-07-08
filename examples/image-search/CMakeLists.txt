project(image-search)

set(CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(usearch
    GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
    GIT_TAG v0.19.3
)
FetchContent_MakeAvailable(usearch)

add_executable(image-search-build
    build.cpp
)

target_link_libraries(image-search-build PRIVATE clip ggml usearch)
target_compile_features(image-search-build PUBLIC cxx_std_17)

add_executable(image-search
    search.cpp
)

target_link_libraries(image-search PRIVATE clip ggml usearch)
target_compile_features(image-search PUBLIC cxx_std_11)

