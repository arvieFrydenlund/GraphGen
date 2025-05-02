import os
import sys
import time
import pydoc
import numpy as np


def get_generator_module(cpp_files=('undirected_graphs.h', 'directed_graphs.h', 'utils.h', 'generator.cpp'),
                         cpp_path='',
                         boost_path='/usr/include/boost/graph/',):
    """
    This will import the C++ module `generator` and may compile it from source if it is not found or is out-of-date.
    """

    def build_module(name, cpp_files, cpp_path, boost_path):
        from os import system  #  -Ofast, don't use this cause numpy warnings
        import sysconfig
        sys_config = sysconfig.get_config_var("EXT_SUFFIX")
        includes = " ".join([f"{cpp_path}{f}" for f in cpp_files]) + " "
        if system(f"g++ --std=c++20 -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared "
                  f"-Wno-sign-compare -Wunused-variable "  # I should fix these instead of suppressing the warnings
                  f"-fPIC $(python3 -m pybind11 --includes) "
                  f"-I{boost_path} "
                  f"-I. {includes} "
                  f"-o generator{sys_config}") != 0:
            print(f"ERROR: Unable to compile `{name}.cpp`.")
            import sys
            sys.exit(1)

    def check_fresh(cpp_files, cpp_path):
        for f in cpp_files:
            if getmtime(generator_module.origin) < os.path.getmtime(f'{cpp_path}{f}'):
                print(f"ERROR: `{f}` is newer than `generator object`. Please recompile.")
                return False
        return True

    try:
        from os.path import getmtime
        import importlib.machinery
        from importlib.util import find_spec

        generator_module = find_spec("generator")
        if generator_module is None:
            raise ModuleNotFoundError
        elif not check_fresh(cpp_files, cpp_path):
            print("C++ module `generator` is out-of-date. Compiling from source...")
            build_module("generator", cpp_files, cpp_path, boost_path)
        import generator
    except ModuleNotFoundError:
        print("C++ module `generator` not found. Compiling from source...")
        build_module("generator", cpp_files, cpp_path, boost_path)
        print(find_spec('generator'))
        import generator

    def help_string():  # displays docstrings from cpp files with print(generator.help_str())
        return pydoc.render_doc(generator, "\nDocstring for %s:")

    setattr(generator, "help_str", help_string)

    return generator


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, suppress=True, )

    print(f'Running from CWD: {os.getcwd()}')
    generator = get_generator_module()
    print("C++ module `generator` loaded.")

    print(f'Random seed is {generator.get_seed()}')
    generator.set_seed(42)
    # generator.set_seed(3172477368)
    print(f'Random seed is {generator.get_seed()} after setting to 42')

    print(generator.help_str())

    # d = generator.erdos_renyi(15, -1.0, 75, 125, False, False, shuffle_edges=False)
    d = generator.euclidian(15, 2, -1, False, False, shuffle_edges=False)

    for k, v in d.items():
        print(f'{k}: {type(v)}')
        if isinstance(v, np.ndarray):
            print(f'\t {k}: {v.shape}, {v.dtype}')
            print(v)
        print()


