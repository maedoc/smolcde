cmake_minimum_required(VERSION 3.10)
project(smolcde C)
set(CMAKE_C_STANDARD 99)

if(NOT MSVC)
    add_compile_options(-Wall -Wextra)
endif()

# Shared library for Python bindings/tests
add_library(smolmaf SHARED maf.c)

# CLI executable - compiled with maf.c statically for portability
add_executable(smolcde main.c maf.c)

if(NOT MSVC)
    target_link_libraries(smolmaf m)
    target_link_libraries(smolcde m)
endif()