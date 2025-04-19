Generate graphs and distances in C++ for use in Python via pybind11


## Build

 `g++ --std=c++20 -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I/usr/include/boost/graph/ -I. undirected_graphs.h directed_graphs.h utils.h generator.cpp -o generator$(python3-config --extension-suffix)`


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