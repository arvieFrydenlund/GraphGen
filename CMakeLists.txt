cmake_minimum_required(VERSION 3.30)
project(GraphGen)

set(CMAKE_CXX_STANDARD 20)

# You need to include python library and pybind11 to the compiler options
# examples:
# -I/usr/include/python3.11 -lpython3.11
# -I/usr/include/python3.10 -lpython3.10
# -I/fs1/home/arvie/PycharmProjects/Virtualenv/Next-Token-Failures/lib/python3.10/site-packages/pybind11/include

add_executable(GraphGen main.cpp
        undirected_graphs.h
        directed_graphs.h,
        generator.cpp)


set(CMAKE_VERBOSE_MAKEFILE ON)


