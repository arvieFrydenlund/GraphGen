# GraphGen — C++ pybind11 extension build
#
# Used both directly (`cmake -S . -B build && cmake --build build`) and via
# scikit-build-core (invoked by `pip install -e .` per pyproject.toml).

cmake_minimum_required(VERSION 3.24)
project(graphgen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Emit compile_commands.json so IDE tooling (VS Code C/C++ IntelliSense,
# clangd, etc.) can find the same include paths CMake uses to build.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Development.Module (not Development.Embed) is what pybind11 needs for a
# Python extension module and is required under scikit-build-core.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# --- The pybind11 module ---------------------------------------------------
pybind11_add_module(generator MODULE
    generator.cpp
    src/generator_config.cpp
    src/csr_graph.cpp
    src/graph_kind.cpp
    src/graph_sampler.cpp
    src/random_utils.cpp
    src/sampled_graph.cpp
    src/worker.cpp
    src/worker_shared_context.cpp
)

target_include_directories(generator PRIVATE
    ${CMAKE_SOURCE_DIR}                             # headers at repo root
    ${CMAKE_SOURCE_DIR}/include                     # graphgen/*.h
    ${CMAKE_SOURCE_DIR}/third_party/mdspan/include  # Kokkos mdspan
)

target_compile_options(generator PRIVATE
    -O3
    -DNDEBUG
    -fno-stack-protector
    -Wall
    -Wpedantic
    -Wno-sign-compare
    -Wno-unused-variable
)

# --- Install rule ----------------------------------------------------------
# scikit-build-core requires an install() call so it knows where to place the
# built module inside the wheel / editable install. Direct
# `cmake --install build` also uses this.
install(TARGETS generator DESTINATION .)

# --- Native C++ tests (doctest) --------------------------------------------
# Only built when configured as a top-level project (i.e. plain
# `cmake -S . -B build`), not when scikit-build-core drives the build to
# produce a wheel. Test executable exercises the pure-C++ pieces where a
# real debugger + fast iteration matter (CsrGraph, algorithms).
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND
    NOT DEFINED SKBUILD)
    enable_testing()

    add_executable(graphgen_tests
        tests_cpp/test_main.cpp
        tests_cpp/test_csr_graph.cpp
        tests_cpp/test_random_utils.cpp
        tests_cpp/test_sampled_graph.cpp
        src/csr_graph.cpp
        src/graph_kind.cpp
        src/random_utils.cpp
        src/sampled_graph.cpp
    )
    target_include_directories(graphgen_tests PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/doctest
    )
    target_compile_options(graphgen_tests PRIVATE
        -O0
        -g
        -Wall
        -Wpedantic
        -Wno-sign-compare
    )
    add_test(NAME graphgen_tests COMMAND graphgen_tests)
endif()


