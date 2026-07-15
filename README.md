
**This a public repo, most of this code is still in a private repo, cleaning up for public release.**


This repository contains:
1. C++ code to generate numpy batches of graphs in C++ and pass them to Python via pybind11
2. Python code to create 'infinite' datastreams for training which sample graphs on the fly [todo move from private repo]
3. Python code for the loss functions for training models (criterion.py) [todo move from private repo]
4. Python code to plot graphs and attention over graphs  [todo move from private repo]

This does not include the training code, but you can use the datastreams and loss functions to train your own models.

### Example of label smoothing different paths and the correct BFS scatchpad

```aiignore
BATCH INDEX: 87
Pos:                                                                                             11    100   101   102   103   4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     700   701   702   703   704   705   706   707   708   709   710   711   712   713   714   715   716   717   718   719   720   721   722   723   724   725   726   727   728   729   730   731   732   733   734   735   736   737   738   739   740   741   742   743   744   745   746   747   748   749   750   751   752   753   754   755   756   757   758   759   760   761   762   763   764   765   766   767   768   769   770   771   772   773   774   775   776   777   778   779   780   781   782   783   784                                                                                                                                                                                             
Src:                                                                                             <s>   /     4     25    ?     25    6     28    32    3     9     3     7     21    18    18    13    13    6     9     28    7     32    3     13    7     25    26    6     32    28    15    3     4     9     24    7     27    21    18    19    16    13    #     D0    4     [     3     ]     D1    3     [     24    9     ]     D2    24    [     7     13    ]     9     [     18    ]     D3    7     [     27    19    28    21    ]     13    [     ]     18    [     16    ]     D4    27    [     ]     19    [     6     ]     28    [     ]     21    [     ]     16    [     ]     D5    6     [     32    ]     D6    32    [     15    26    ]     D7    15    [     ]     26    [     25    ]     =     4     3     24    7     19    6     32    26    25    .     </s>                                                                                                                                                                                            
                                                                                                 <s>   /     4     25    ?     26    32    6     15    4     24    9     27    7     19    16    19    21    19    18    7     24    26    24    24    19    25    26    6     32    28    15    3     4     9     24    7     27    21    18    19    16    13    #     D0    4     [     3     ]     D1    3     [     24    9     ]     D2    24    [     7     13    ]     9     [     18    ]     D3    7     [     27    19    28    21    ]     13    [     ]     18    [     16    ]     D4    27    [     ]     19    [     6     ]     28    [     ]     21    [     ]     16    [     ]     D5    6     [     32    ]     D6    32    [     15    26    ]     D7    15    [     ]     26    [     25    ]     =     4     3     24    7     19    6     32    26    25    .     </s>                                                                                                                                                                                            
Tgt:                                                                                                                                                                                                                                                                                                                                                               #     D0    4     [     3     ]     D1    3     [     24    9     ]     D2    24    [     7     13    ]     9     [     18    ]     D3    7     [     27    19    28    21    ]     13    [     ]     18    [     16    ]     D4    27    [     ]     19    [     6     ]     28    [     ]     21    [     ]     16    [     ]     D5    6     [     32    ]     D6    32    [     15    26    ]     D7    15    [     ]     26    [     25    ]     =     4     3     24    7     19    6     32    26    25    .     </s>                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                         9                                   13                                                          19    28    21                                                                                                                                                                                                                26                                                                                  9     13    28                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         28    21                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         21                                                                                                                                                                                                                                                                   
```


TODO make image

DFS

```aiignore
BATCH INDEX: 2
Pos:   11    100   101   102   103   4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     4     700   701   702   703   704   705   706   707   708   709   710   711   712   713   714   715   716   717   718   719   720   721   722   723   724   725   726   727   728   729   730   731   732   733   734   735   736   737   738   739   740   741   742   743   744   745   746   747   748   749   750   751   752   753   754   755   756   757   758   759   
Src:   <s>   /     8     1     ?     6     1     10    31    26    26    23    23    4     3     23    4     8     3     31    22    26    19    30    31    28    27    6     28    1     0     10    25    31    19    26    11    9     23    22    4     3     30    8     14    27    #     D0    8     4     D1    4     30    D2    30    26    D3    26    9     D4    9     28    D5    28    31    D6    31    14    D6    31    19    D7    19    3     D7    19    23    D8    23    10    D9    10    25    D10   25    22    D8    23    0     D9    0     1     =     8     4     30    26    9     28    31    19    23    0     1     .     </s>  
       <s>   /     8     1     ?     28    0     25    19    11    9     0     22    26    19    10    30    4     31    28    25    30    23    9     14    9     6     6     28    1     0     10    25    31    19    26    11    9     23    22    4     3     30    8     14    27    #     D0    8     4     D1    4     30    D2    30    26    D3    26    9     D4    9     28    D5    28    31    D6    31    14    D6    31    19    D7    19    3     D7    19    23    D8    23    10    D9    10    25    D10   25    22    D8    23    0     D9    0     1     =     8     4     30    26    9     28    31    19    23    0     1     .     </s>  
Tgt:                                                                                                                                                                                                                                                                                       #     D0    8     4     D1    4     30    D2    30    26    D3    26    9     D4    9     28    D5    28    31    D6    31    14    D6    31    19    D7    19    3     D7    19    23    D8    23    10    D9    10    25    D10   25    22    D8    23    0     D9    0     1     =     8     4     30    26    9     28    31    19    23    0     1     .     </s>  
                                                                                                                                                                                                                                                                                                                               26                9                 11                                  6                 19                3                 23                                  0                                                     22                                        26                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                         3                                                                       22                                                                                                                                                                
Idx:   0     1     2     3     4     5     6     7     8     9     10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34    35    36    37    38    39    40    41    42    43    44    45    46    47    48    49    50    51    52    53    54    55    56    57    58    59    60    61    62    63    64    65    66    67    68    69    70    71    72    73    74    75    76    77    78    79    80    81    82    83    84    85    86    87    88    89    90    91    92    93    94    95    96    97    98    99    100   101   102   103   104   105   

```


#### C++ code


######  TODO graph processing helpers

Make functions which take in an arbitrary edge list and get tokenization, this is for when the graph structure is known ahead of time.
This will be useful for real world datasets where the graphs are pre-defined.

#### Python code

## Build

The C++ extension is built with CMake, driven by
[scikit-build-core](https://scikit-build-core.readthedocs.io/). No manual
`g++` invocation, no `python3-config`, no per-machine include paths.

### Prerequisites

* Python 3.10 (`.venv` set up with [uv](https://github.com/astral-sh/uv) or plain `python -m venv`)
* Boost headers
    * macOS: `bash install_mac_boost.sh` (or `brew install boost`)
    * Linux: `sudo apt install libboost-all-dev`
* A C++20 compiler (clang on macOS, g++ on Linux)

CMake, Ninja, pybind11 and scikit-build-core are pinned as runtime deps in
[pyproject.toml](pyproject.toml), so they get installed into the venv
automatically -- you don't need them on the system.

> Note: scikit-build-core drives CMake, and CMake in turn uses
> [Ninja](https://ninja-build.org/) as its build backend to run the
> compile/link commands.

### First install

```bash
bash install.sh
```

That's it. Under the hood the script:

1. Picks `.venv/bin/python` (or `$PYTHON` if set, or `python3`).
2. Installs the build deps (`scikit-build-core`, `cmake`, `ninja`,
   `pybind11`) into the venv so `--no-build-isolation` works.
3. Runs `pip install -e . --no-build-isolation`, which invokes CMake +
   Ninja to compile [generator.cpp](generator.cpp) into
   `generator.cpython-310-*.so` and drops a scikit-build-core
   editable-install shim into `site-packages`.

### Rebuilding after C++ edits

Just re-import. `[tool.scikit-build.editable].rebuild = true` in
[pyproject.toml](pyproject.toml) makes the editable-install shim run
`cmake --build` transparently every time you `import generator`, so any
change to `generator.cpp` / the header files / `CMakeLists.txt` picks up
on the next `python` invocation with no explicit rebuild step.

There is also a Python fallback: `get_generator_module()` in
[get_generator_module.py](get_generator_module.py) shells out to
`pip install -e .` if `import generator` fails outright (e.g. fresh
checkout, `.so` deleted).

### Verifying a build

Minimal smoke test -- doesn't touch dictionaries, just proves the module
compiled and the FFI works:

```bash
source .venv/bin/activate
python -c "
from get_generator_module import get_generator_module
get_generator_module()
import generator
generator.set_seed(1234)
assert generator.get_seed() == 1234
print('OK')
"
```

### Manual CMake build (optional, IDE-friendly)

If you want to compile without pip in the loop -- e.g. from CLion:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The `.so` lands in `build/`. You'll need `pybind11` findable on
`CMAKE_PREFIX_PATH` (easiest: activate the venv and run `cmake` from
there).

### Vendored dependency

[Kokkos mdspan](https://github.com/kokkos/mdspan) is vendored under
`third_party/mdspan/` (BSD-3-Clause). See
`third_party/mdspan/VENDORED.md` for source and refresh instructions.


## Citation

Code from [https://github.com/asaparov/learning_to_search](https://github.com/asaparov/learning_to_search) was a great help in figuring out some things, like using different seeding across python threads and pybind.

from paper:
> @inproceedings{
TransformersStruggleToSearch,
title={Transformers Struggle to Learn to Search},
author={Abulhair Saparov and Srushti Pawar and Shreyas Pimpalgaonkar and Nitish Joshi and Richard Yuanzhe Pang and Vishakh Padmakumar and Seyed Mehran Kazemi and Najoung Kim and He He},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=qFVVBzXxR2V}
}

