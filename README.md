

# HEAT MAPS [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) ![C++](https://img.shields.io/badge/C++-Solutions-red.svg?style=flat&logo=c++) 

Set of tools to generate heat maps from standard bild formats. The algorithm is based on the OpenCV modul, which was implemented in C++ but could also be compiled in cython to be imported und used in python.  
## Dependencies

 * [numpy]
 * [cython]
 * [OpenCV]

## Setup

This implementation supports python 3.6+.

To use it, you should clone the repository and place it on your local directory.
Now, go into the folder

    "/HMG

The implementation can be compiled in C++ with the file main.cpp
    
    g++ heatMap.cpp -o heatmap `pkg-config --cflags --libs opencv` 

Or, it could be compiled with cython. In that case, perform the following commands on the terminal:

    python3.? setup.py build_ext --inplace