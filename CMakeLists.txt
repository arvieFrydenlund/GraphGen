# GraphGen — C++ pybind11 extension build
#
# Used both directly (`cmake -S . -B build && cmake --build build`) and via
# scikit-build-core (invoked by `pip install -e .` per pyproject.toml).
#
# Step 1a of PLAN.md's Monitored execution plan: replaces the ad-hoc
# install.sh + setup.py chain without touching any C++ source. V1 sources
# still build the same `generator` module.

cmake_minimum_required(VERSION 3.24)
project(graphgen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Development.Module (not Development.Embed) is what pybind11 needs for a
# Python extension module and is required under scikit-build-core.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# --- Boost -----------------------------------------------------------------
# Matches V1's install.sh / setup.py detection: pick up Homebrew boost on
# macOS when present, fall back to system paths otherwise. Goes away with
# R5 Level 2 (see PLAN.md); until then it is how V1 headers compile.
if(APPLE)
    execute_process(
        COMMAND brew --prefix boost
        OUTPUT_VARIABLE _brew_boost_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _brew_boost_rc
        ERROR_QUIET
    )
    if(_brew_boost_rc EQUAL 0 AND EXISTS "${_brew_boost_prefix}/include/boost")
        set(BOOST_ROOT "${_brew_boost_prefix}" CACHE PATH "Homebrew Boost prefix")
        set(Boost_NO_SYSTEM_PATHS ON)
    endif()
endif()
find_package(Boost REQUIRED)

# --- The pybind11 module ---------------------------------------------------
pybind11_add_module(generator MODULE generator.cpp)

target_include_directories(generator PRIVATE
    ${CMAKE_SOURCE_DIR}                             # V1 headers at repo root
    ${CMAKE_SOURCE_DIR}/third_party/mdspan/include  # Kokkos mdspan (R12)
    ${Boost_INCLUDE_DIRS}
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


