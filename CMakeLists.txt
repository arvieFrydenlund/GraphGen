cmake_minimum_required(VERSION 3.30)
project(GraphGen)

set(CMAKE_CXX_STANDARD 20)

find_library(PYBIND11_INCLUDE_DIR
        NAMES pybind11
        PATHS /home/arvie/PycharmProjects/Virtualenv/learning_to_search/lib/python3.11/site-packages/pybind11/include/
        NO_DEFAULT_PATH)
if(NOT PYBIND11_INCLUDE_DIR)
    message(FATAL_ERROR "pybind11 include directory not found")
else()
    message(STATUS "pybind11 include directory found: ${PYBIND11_INCLUDE_DIR}")
endif()

gcc -I/${PYBIND11_INCLUDE_DIR}

add_executable(GraphGen main.cpp
        undirected_graphs.h
        directed_graphs.h
        data_gen.h
        py_bindings.h)


target_include_directories(GraphGen INTERFACE ${PYBIND11_INCLUDE_DIR})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(GraphGen ${PYTHON_LIBRARIES})



# -I /home/arvie/PycharmProjects/Virtualenv/learning_to_search/lib/python3.11/site-packages/pybind11/include
# target_include_directories(GraphGen PRIVATE /home/arvie/PycharmProjects/Virtualenv/learning_to_search/lib/python3.11/site-packages/pybind11/include/pybind11)
# link_directories(/home/arvie/PycharmProjects/Virtualenv/learning_to_search/lib/python3.11/site-packages/pybind11/include/)
#DPython3_EXECUTABLE(/home/arvie/PycharmProjects/Virtualenv/learning_to_search/bin)

