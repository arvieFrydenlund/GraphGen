import time
import numpy as np

np.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, suppress=True,)


"""
Minimal example of pybind compiling
"""

def build_module(name):
    from os import system
    #  f"-I. /usr/include/boost/ "
    if system(f"g++ -Ofast -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I. {name}.cpp -o {name}$(python3-config --extension-suffix)") != 0:
        print(f"ERROR: Unable to compile `{name}.cpp`.")
        import sys
        sys.exit(1)


try:
    name = "generator_test"
    from os.path import getmtime
    from importlib.util import find_spec
    generator_module = find_spec("generator_test")
    if generator_module == None:
        raise ModuleNotFoundError
    elif getmtime(generator_module.origin) < getmtime(f'{name}.cpp'):
        print("C++ module `generator` is out-of-date. Compiling from source...")
        build_module(name)
    import generator_test
except ModuleNotFoundError:
    print("C++ module `generator` not found. Compiling from source...")
    build_module(name)
    import generator_test
print("C++ module `generator` loaded.")

print(generator_test.add(2, 5))


