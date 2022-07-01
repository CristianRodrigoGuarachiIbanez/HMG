#distutils: language = c++
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.utility cimport pair
cimport numpy as np
import numpy as np

# For cv::Mat usage
cdef extern from "opencv2/core/core.hpp":
  cdef int  CV_WINDOW_AUTOSIZE
  cdef int CV_8UC3
  cdef int CV_8UC1
  cdef int CV_32FC1
  cdef int CV_8U
  cdef int CV_32F

cdef extern from "opencv2/core/core.hpp" namespace "cv":
  cdef cppclass Mat:
    Mat() except +
    void create(int, int, int)
    void* data
    int rows
    int cols
    int channels()
    int depth()
    size_t elemSize()

# For Buffer usage
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int readonly, int infoflags)
    enum:
        PyBUF_FULL_RO

cdef extern from "../heatMap.h" namespace "heatmap":
    cdef cppclass HeatMaps:
        HeatMaps(const char*filename, const char *output, int rows, int cols, int x, int w, int y, int h ) except +
        inline Mat getHeatMap()
        inline Mat getBlendedImg()
        Mat heatMap;
        Mat output;
        inline Mat  openMat(const char*filename)
        void create_heatmap(Mat&img, Mat&mask, int rows, int cols, int x, int w, int y, int h );
        void resized(Mat&img, int rows, int cols);
        void blend_images(Mat&mask,Mat&image, int x, int w, int y, int h);