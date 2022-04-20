from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# setup(
#     ext_modules=[
#         Extension("my_module", ["my_module.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )
#
# # Or, if you use cythonize() to make the ext_modules list,
# # include_dirs can be passed to setup()
ext = Extension(name="visualizer", sources=["GlimpseVisualizer.pyx"])
setup(
    ext_modules=cythonize(ext),
    include_dirs=[numpy.get_include()]
)