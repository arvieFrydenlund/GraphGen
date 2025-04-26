import time
import pydoc
import numpy as np

# np.finfo(np.dtype("float32"))  # gets rid of warnings, hope they aint important
# np.finfo(np.dtype("float64"))
np.set_printoptions(threshold=np.inf, edgeitems=10, linewidth=np.inf, precision=2, suppress=True,)


def build_module(name):
    from os import system  #  -Ofast, don't use this cause numpy warnings
    if system(f"g++ --std=c++20 -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared "
              f"-fPIC $(python3 -m pybind11 --includes) "
              f"-I/usr/include/boost/graph/ "
              f"-I. undirected_graphs.h directed_graphs.h utils.h generator.cpp "
              f"-o generator$(python3-config --extension-suffix)") != 0:
        print(f"ERROR: Unable to compile `{name}.cpp`.")
        import sys
        sys.exit(1)

try:
    from os.path import getmtime
    from importlib.util import find_spec
    generator_module = find_spec("generator")
    if generator_module is None:
        raise ModuleNotFoundError
    elif getmtime(generator_module.origin) < getmtime(f'{"generator"}.cpp'):
        print("C++ module `generator` is out-of-date. Compiling from source...")
        build_module("generator")
    import generator
except ModuleNotFoundError:
    print("C++ module `generator` not found. Compiling from source...")
    build_module("generator")
    import generator

print("C++ module `generator` loaded.")


print(f'Random seed is {generator.get_seed()}')
generator.set_seed(42)
# generator.set_seed(3172477368)
print(f'Random seed is {generator.get_seed()} after setting to 42')

h = pydoc.render_doc(generator, "Help on %s")
print(h)

# d = generator.erdos_renyi(15, -1.0, 75, 125, False, False, shuffle_edges=False)
d = generator.euclidian(15, 2, -1, False, False, shuffle_edges=False)

for k, v in d.items():
    print(f'{k}: {type(v)}')
    if isinstance(v, np.ndarray):
        print(f'\t {k}: {v.shape}, {v.dtype}')
        print(v)
    print()
print('\n\nTesting N generator')
print(f'Test size is {generator.get_test_size()}')

d_n = generator.euclidian_n(50, 56, 2, -1, is_causal=True, shuffle_nodes=True, min_vocab=4, max_vocab=60, shuffle_edges=True, batch_size=8)

for k, v in d_n.items():
    print(f'{k}: {type(v)}', end='')
    if isinstance(v, np.ndarray):
        print(f'\t {k}: {v.shape}, {v.dtype}')
    else:
        print()


print(f'Settings hashes')
hashes = d_n['hashes']
generator.set_test_hashes(hashes)
print(f'Test size is {generator.get_test_size()}')
print(f'Is in same: {generator.is_in_test(hashes)}')
hashes[1] = 0  # fake a different hash
print(f'Is in with 2nd faked: {generator.is_in_test(hashes)}')

d_n = generator.euclidian_n(50, 55, 2, -1, is_causal=True, min_vocab=4, max_vocab=59, shuffle_edges=True, batch_size=8)
print(f"Is in with new hashes: {generator.is_in_test(d_n['hashes'])}")


print('\n\nSetting dictionary')
fake_dict = {str(i): i for i in range(12)}
generator.set_dictionary(fake_dict)


print('\n\nDone')