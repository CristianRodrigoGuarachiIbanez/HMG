from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import sys
import os
import glob
#import pkgconfig

#var = pkgconfig.variables("opencv")
lib_folder = os.path.join("/usr", 'lib', "x86_64-linux-gnu")
#lib_folder = list(var.values()) #os.path.join(sys.prefix, "lib")
# Find opencv libraries in lib_folder

cvlibs = list()
for file in glob.glob(os.path.join(lib_folder, 'libopencv_*')):
    cvlibs.append(file.split('.')[0])

#cvlibs = pkgconfig.libs("opencv") #
#cvlibs = list(set(cvlibs))
print( cvlibs)
cvlibs = ['-L{}'.format(lib_folder)] + ['opencv_{}'.format(lib.split(os.path.sep)[-1].split('libopencv_')[-1]) for lib in cvlibs]


print("LIBS",cvlibs, os.path.join("usr" ,"include", "opencv2"))
print("FOLDER:", lib_folder, sys.prefix)
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(Extension("heatmap",
                                    sources=["heatmap.pyx", "heatMap.cpp"],
                                    language="c++",
                                    include_dirs=[numpy.get_include(),
                                                  os.path.join("/usr", 'include', 'opencv'),
                                                 ],
                                    library_dirs=[lib_folder],
                                    libraries=cvlibs,
                                    )
                          )
)

#ldconfig -p | grep opencv
