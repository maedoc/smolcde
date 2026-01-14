cmake_minimum_required(VERSION 3.10)
project(smolcde C)
set(CMAKE_C_STANDARD 99)

if(MSVC)
    # /W4 is the standard for "Wall", /fp:fast for fast-math, /O2 for O3
    add_compile_options(/W4 /O2 /fp:fast)
    # Note: No "native" flag; you must pick a specific SIMD set
    add_compile_options(/arch:AVX2) 
else()
    add_compile_options(-Wall -Wextra -O3 -ffast-math)
    option(ENABLE_NATIVE_ARCH "Enable -march=native" ON)
    if(ENABLE_NATIVE_ARCH)
        add_compile_options(-march=native)
    endif()
endif()

# Shared library for Python bindings/tests
add_library(smolmaf SHARED maf.c)

# CLI executable - compiled with maf.c statically for portability
add_executable(smolcde main.c maf.c)

if(NOT MSVC)
    target_link_libraries(smolmaf m)
    target_link_libraries(smolcde m)
endif()
