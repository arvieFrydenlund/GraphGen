
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


#### C++ code

#### Python code

## Build

[todo]  this is outdated

 `g++ --std=c++20 -DNDEBUG -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I/usr/include/boost/graph/ -I. undirected_graphs.h directed_graphs.h utils.h generator.cpp -o generator$(python3-config --extension-suffix)`

Note this works if using the primary system python, if you have multiple versions of python installed [see here where python3-config --extension-suffix fails](https://stackoverflow.com/questions/77112605/what-is-the-prefered-way-of-generating-extension-module-filename-suffix-in-virtu) 

Note to self: running with `-fsanitize=address -g` helps with debugging.  

You need to include python library and pybind11 to the compiler options, for Clion include
examples:
* -I/usr/include/python3.11 -lpython3.11
* -I/usr/include/python3.10 -lpython3.10
* -I/home/arvie/PycharmProjects/Virtualenv/Next-Token-Failures/lib/python3.10/site-packages/pybind11/include


## Citation

Code from [https://github.com/asaparov/learning_to_search](https://github.com/asaparov/learning_to_search) was a great help in figuring out some things, like using different seeding across python threads.

from paper:
> @inproceedings{
TransformersStruggleToSearch,
title={Transformers Struggle to Learn to Search},
author={Abulhair Saparov and Srushti Pawar and Shreyas Pimpalgaonkar and Nitish Joshi and Richard Yuanzhe Pang and Vishakh Padmakumar and Seyed Mehran Kazemi and Najoung Kim and He He},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=qFVVBzXxR2V}
}

