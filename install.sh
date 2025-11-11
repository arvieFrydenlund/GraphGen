#!/bin/bash

g++ --std=c++20 -Ofast -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared -Wno-sign-compare -Wunused-variable \
  -fPIC $(python3 -m pybind11 --includes) \
  -I/usr/include/boost/graph/ \
  -I. undirected_graphs.h directed_graphs.h utils.h matrix.h graph_tokenizer.h tasks.h scratch_pads.h instance.h generator.cpp -o generator$(python3-config --extension-suffix)

python setup.py build
python setup.py install
