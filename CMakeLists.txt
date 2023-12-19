cmake_minimum_required(VERSION 3.12)
project("CLIP.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CLIP_STANDALONE ON)
else()
    set(CLIP_STANDALONE OFF)
endif()

#
# Option list
#

# general
option(CLIP_STATIC                 "CLIP: static link libraries"                          OFF)
option(CLIP_BUILD_TESTS             "CLIP: build tests"                                    ${CLIP_STANDALONE})
option(CLIP_BUILD_EXAMPLES         "CLIP: build examples"                                 ${CLIP_STANDALONE})
option(CLIP_BUILD_IMAGE_SEARCH     "CLIP: build image-search"                             OFF)
option(CLIP_NATIVE                 "CLIP: enable -march=native flag"                      ON)
option(CLIP_LTO                    "CLIP: enable link time optimization"                  OFF)

# debug
option(CLIP_ALL_WARNINGS           "CLIP: enable all compiler warnings"                   OFF)
option(CLIP_ALL_WARNINGS_3RD_PARTY "CLIP: enable all compiler warnings in 3rd party libs" OFF)
option(CLIP_GPROF                  "CLIP: enable gprof"                                   OFF)

# sanitizers
option(CLIP_SANITIZE_THREAD        "CLIP: enable thread sanitizer"                        OFF)
option(CLIP_SANITIZE_ADDRESS       "CLIP: enable address sanitizer"                       OFF)
option(CLIP_SANITIZE_UNDEFINED     "CLIP: enable undefined sanitizer"                     OFF)

# instruction set specific
option(CLIP_AVX                    "CLIP: enable AVX"                                     ON)
option(CLIP_AVX2                   "CLIP: enable AVX2"                                    ON)
option(CLIP_FMA                    "CLIP: enable FMA"                                     ON)
option(CLIP_AVX512                 "clip: enable AVX512"                                  OFF)
option(CLIP_AVX512_VBMI            "clip: enable AVX512-VBMI"                             OFF)
option(CLIP_AVX512_VNNI            "clip: enable AVX512-VNNI"                             OFF)
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(CLIP_F16C               "clip: enable F16C"                                    ON)
endif()


# 3rd party libs
option(CLIP_ACCELERATE             "CLIP: enable Accelerate framework"                    ON)
option(CLIP_OPENBLAS               "CLIP: use OpenBLAS"                                   OFF)


#
# Compile flags
#

if (CLIP_STATIC)
    add_compile_definitions(STB_IMAGE_STATIC)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT MSVC)
    if (CLIP_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (CLIP_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (CLIP_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND CLIP_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(CLIP_EXTRA_LIBS ${CLIP_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()
if (CLIP_OPENBLAS)
    if (CLIP_STATIC)
        set(BLA_STATIC ON)
    endif()

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "OpenBLAS found")

        add_compile_definitions(GGML_USE_OPENBLAS)
        add_link_options(${BLAS_LIBRARIES})
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

if (CLIP_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wno-format
            -Wcast-qual
            -Wdouble-promotion
            -Wshadow
            -Wstrict-prototypes
            -Wpointer-arith
            -Wno-unused-function
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wno-format
            -Wcast-qual
            -Wno-unused-function
        )
    else()
        # todo : msvc
    endif()

    add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (CLIP_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()



# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (CLIP_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (CLIP_GPROF)
        add_compile_options(-pg)
    endif()
    if (CLIP_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-mcpu=native)
        endif()
        # TODO: armv6,7,8 version specific flags
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (CLIP_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (CLIP_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (CLIP_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
        elseif (CLIP_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (CLIP_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        if (CLIP_F16C)
            add_compile_options(-mf16c)
        endif()
        if (CLIP_FMA)
            add_compile_options(-mfma)
        endif()
        if (CLIP_AVX)
            add_compile_options(-mavx)
        endif()
        if (CLIP_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (CLIP_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
        if (CLIP_AVX512_VBMI)
            add_compile_options(-mavx512vbmi)
        endif()
        if (CLIP_AVX512_VNNI)
            add_compile_options(-mavx512vnni)
        endif()
    endif()
else()
    # TODO: support PowerPC
    message(STATUS "Unknown architecture")
endif()



#
# Build libraries
#

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_subdirectory(ggml)

add_library(clip
            clip.cpp
            clip.h)

target_include_directories(clip PUBLIC .)
target_compile_features(clip PUBLIC cxx_std_11)
target_link_libraries(clip PRIVATE ggml ${CLIP_EXTRA_LIBS})

if (BUILD_SHARED_LIBS)
    set(CLIP_BUILD_EXAMPLES OFF)
    set(CLIP_BUILD_TESTS OFF)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(clip PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(clip PRIVATE CLIP_SHARED CLIP_BUILD)
    set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR})
    install(TARGETS ggml clip
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/examples/python_bindings/clip_cpp)
endif()

# add_subdirectory(models)

if (CLIP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (CLIP_BUILD_TESTS)
    add_subdirectory(tests)
endif()
