cmake_minimum_required(VERSION 3.30)
project(GraphGen)

set(CMAKE_CXX_STANDARD 20)

add_executable(GraphGen main.cpp
        cpp_graph.h)
