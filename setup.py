from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "example",  # Module name
        [],
        headers=["undirected_graphs.h ", "directed_graphs.h", "py_bindings.h"],  # Source files
        include_dirs=[pybind11.get_include()],  # Include pybind11 headers
        language="c++20",  # Specify C++ as the language
    )
]

setup(
    name="example",
    version="0.1",
    ext_modules=ext_modules,
)