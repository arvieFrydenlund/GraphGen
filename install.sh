#!/bin/bash

g++ --std=c++20 -Ofast -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I/usr/include/boost/graph/ -I. undirected_graphs.h directed_graphs.h generator.cpp -o generator$(python3-config --extension-suffix)

python setup.py build
python setup.py install
