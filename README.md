Generate graphs and distances in C++ for use in Python via pybind11


## Code

This repository contains:
1. C++ code to generate batches of graphs in C++ and pass them to Python via pybind11
2. Python code to create datastreams for training and testing
3. Python code for the loss functions for training models (criterion.py)
4. Python code to plot graphs

This does not include the training code, but you can use the datastreams and loss functions to train your own models.
Full code will not be provided for some time, but the setup is very easy to reproduce given the code provided here.


#### C++ code

#### Python code



## Build

 `g++ --std=c++20 -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I/usr/include/boost/graph/ -I. undirected_graphs.h directed_graphs.h utils.h generator.cpp -o generator$(python3-config --extension-suffix)`

Note this works if using the primary system python, if you have multiple versions of python installed [see here where python3-config --extension-suffix fails](https://stackoverflow.com/questions/77112605/what-is-the-prefered-way-of-generating-extension-module-filename-suffix-in-virtu) 

Note to self: running with `-fsanitize=address -g` helps with debugging.  

You need to include python library and pybind11 to the compiler options, for Clion include
examples:
* -I/usr/include/python3.11 -lpython3.11
* -I/usr/include/python3.10 -lpython3.10
* -I/home/arvie/PycharmProjects/Virtualenv/Next-Token-Failures/lib/python3.10/site-packages/pybind11/include


## Citation

Code based on C++ code from [https://github.com/asaparov/learning_to_search](https://github.com/asaparov/learning_to_search)

from paper:
> @inproceedings{
TransformersStruggleToSearch,
title={Transformers Struggle to Learn to Search},
author={Abulhair Saparov and Srushti Pawar and Shreyas Pimpalgaonkar and Nitish Joshi and Richard Yuanzhe Pang and Vishakh Padmakumar and Seyed Mehran Kazemi and Najoung Kim and He He},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=qFVVBzXxR2V}
}

Please cite if you use this code in your research.