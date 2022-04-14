#distutils: language = c++
from libs.heatmap cimport *
from libcpp.vector cimport vector
from cython cimport boundscheck, wraparound, cdivision
from numpy import uint8, asarray, ndarray, zeros, float32
ctypedef unsigned char uchar
cdef class HEATMAPS:
    cdef:
        vector[Mat] heatmaps
        vector[Mat] blended
    def __cinit__(self, const char* mask_path, list masks, const char*path, list directories, int rows, int cols, int x, int w, int y, int h ):
        self.display(mask_path, masks, path, directories, rows, cols, x, w, y, h)
    cdef void display(self, const char*mask_path, list masks, const char*path, list directories, int rows, int cols, int x, int w, int y, int h ):
        cdef:
            size_t i, dir
            HeatMaps * map
        dir = len(directories)
        for i in range(dir):
            print("directory -> ", path +directories[i], mask_path+masks[i])
            map = new HeatMaps(path+directories[i], mask_path+masks[i], rows, cols, x, w, y, h)
            self.heatmaps.push_back(map.getHeatMap())
            self.blended.push_back(map.getBlendedImg())
            del map
        if(len(directories) == self.heatmaps.size()):
            print("size compatible")
    @boundscheck(True)
    @wraparound(True)
    @cdivision(True)
    cdef inline object Mat2np(self, Mat&m):
        # Create buffer to transfer data from m.data
        cdef Py_buffer buf_info
        # Define the size / len of data
        cdef size_t len = m.rows*m.cols*m.elemSize()  #m.channels()*sizeof(CV_8UC3)
        # Fill buffer
        PyBuffer_FillInfo(&buf_info, NULL, m.data, len, 1, PyBUF_FULL_RO)

        # Get Pyobject from buffer data
        Pydata  = PyMemoryView_FromBuffer(&buf_info)

        # Create ndarray with data
        if m.channels() >1 :
            shape_array = (m.rows, m.cols, m.channels())
        else:
            shape_array = (m.rows, m.cols)

        if m.depth() == CV_32F :
            ary = ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=float32)
        else :
            #8-bit image
            ary = ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=uint8)
        return asarray(ary, dtype=uint8)

    cdef uchar[:,:,:,:] recover_images(self, bint hm, int c, int r):
        cdef:
            size_t i, j, k, n, size
            int rows, cols
            Mat  img
            #uchar[:,:,:,:] output
        output = []
        if (hm is True):
            size = self.heatmaps.size()
            rows = self.heatmaps[0].rows
            cols = self.heatmaps[0].cols
            for i in range(size):
                img = self.heatmaps[i]
                arr = self.Mat2np(img)
                print("SHAPE ->",arr.shape, "rows", rows, "cols", cols)
                if(arr.shape[0]==rows and arr.shape[1]==cols):
                    output.append(arr)
        else:
            size = self.blended.size()
            rows = r
            cols = c
            for i in range(size):
                img = self.blended[i]
                arr = self.Mat2np(img)
                print("SHAPE ->",arr.shape, "rows", rows, "cols", cols)
                if(arr.shape[0]==rows and arr.shape[1]==cols):
                    output.append(arr)

        return asarray(output, dtype=uint8)
    def get_heatmaps(self, rows=120, cols=160, hm=True):
        return asarray(self.recover_images(hm, cols, rows),dtype=uint8)