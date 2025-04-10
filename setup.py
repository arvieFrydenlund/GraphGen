from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "generator",  # Module name
        ["generator.cpp"],  # Source files
        # headers=["undirected_graphs.h ", "directed_graphs.h"],
        include_dirs=[pybind11.get_include()],  # Include pybind11 headers
        language="c++20",  # Specify C++ as the language
    )
]

setup(
    name="generator",
    version="0.1",
    ext_modules=ext_modules,
)