cmake_minimum_required(VERSION 3.30)
project(GraphGen)

set(CMAKE_CXX_STANDARD 20)

add_executable(GraphGen main.cpp
        py_bindings.h
        undirected_graphs.h
        directed_graphs.h)


# -I/usr/include/python3.11 -lpython3.11
# -I/usr/include/python3.10 -lpython3.10
# -I/fs1/home/arvie/PycharmProjects/Virtualenv/Next-Token-Failures/lib/python3.10/site-packages/pybind11/include

#set(DPYTHON_INCLUDE_DIR, $(python3.11 -c "import sysconfig; print(sysconfig.get_path('include'))"))
#set(DPYTHON_LIBRARY, $(python.11 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"))


# find python3.11
# find_package(Python3 3.11 REQUIRED COMPONENTS Development)
# if (NOT Python3_FOUND)
#    message(FATAL_ERROR "Python3 not found")
#else()
#    message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
#    message(STATUS "Python3 include directory: ${Python3_INCLUDE_DIRS}")
#    message(STATUS "Python3 library: ${Python3_LIBRARIES}")
#    message(STATUS "Python3 version: ${Python3_VERSION}")
#    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
# endif ()



# target_include_directories(GraphGen INTERFACE ${PYBIND11_INCLUDE_DIR})

# -I /home/arvie/PycharmProjects/Virtualenv/learning_to_search/lib/python3.11/site-packages/pybind11/include
# target_include_directories(GraphGen PRIVATE /home/arvie/PycharmProjects/Virtualenv/learning_to_search/lib/python3.11/site-packages/pybind11/include/pybind11)
# link_directories(/home/arvie/PycharmProjects/Virtualenv/learning_to_search/lib/python3.11/site-packages/pybind11/include/)
#DPython3_EXECUTABLE(/home/arvie/PycharmProjects/Virtualenv/learning_to_search/bin)

