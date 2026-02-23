cmake_minimum_required(VERSION 3.30)
project(GraphGen)

# source /h/arvie/PycharmProjects/Virtualenv/next_token_3.12/bin/activate

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python interpreter: ${Python3_EXECUTABLE}")

# You need to include python library and pybind11 to the compiler options
# examples:
# -I/usr/include/python3.12 -lpython3.12
# -I/usr/include/python3.10 -lpython3.10
# -I/h/arvie/PycharmProjects/Virtualenv/next_token_3.12/lib/python3.12/site-packages/pybind11/include

add_executable(GraphGen main.cpp
        undirected_graphs.h
        directed_graphs.h,
        graph_wrapper.h,
        args.h,
        utils.h,
        matrix.h,
        graph_tokenizer.h,
        tasks.h,
        scratch_pads.h,
        instance.h,
        generator.cpp,
        tests.h)


set(CMAKE_VERBOSE_MAKEFILE ON)


