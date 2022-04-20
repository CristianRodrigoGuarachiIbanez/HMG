# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stdlib cimport malloc,free
from libcpp.string cimport string
from libcpp.utility cimport pair
from numpy import uint8, zeros,asarray,float64, ones
from cython cimport boundscheck, wraparound, cdivision
from cython.parallel import prange, parallel

ctypedef unsigned char uchar
cdef struct Points:
    int start
    int end

cdef class GlimpseConstructor:
    cdef:
        uchar **** heat_maps
        size_t dim1,dim2,dim3,dim4,images

    def __cinit__(self, uchar[:,:,:,:,:] features, uchar[:,:,:] labels, double[:,:,:] positions):
        self.setFeatureDimensions(features)
        self.heat_maps = <unsigned char****>malloc(labels.shape[2]*sizeof(unsigned char***))
        self.setOutputDimensions(self.heat_maps, labels, positions)
        self.create_heatmaps(labels, positions)
        if(self.heat_maps==NULL):
            raise MemoryError()
    def __deallocate__(self):
        free(self.heat_maps)
    cdef void setFeatureDimensions(self,uchar[:,:,:,:,:] features ):
        if(features.shape[4]>99):
            self.dim3 = features.shape[3] # 120
            self.dim4 = features.shape[4] # 160
        else:
            self.dim3 = features.shape[2] # 120
            self.dim4 = features.shape[3] # 160
    @boundscheck(False)
    @wraparound(False)
    @cdivision(False)
    cdef void setOutputDimensions(self, uchar****&features, uchar[:,:,:] labels, double[:,:,:] positions):
        self.images = labels.shape[0] #N
        self.dim1= labels.shape[2] # 6
        self.dim2 = positions.shape[1] # 11
        self.populate_heatmaps(features, self.dim1, self.dim2, self.dim3, self.dim4)
    @boundscheck(False)
    @wraparound(False)
    @cdivision(False)
    cdef void populate_heatmaps(self, uchar****&heat_map, size_t dim, size_t dim1, size_t rows, size_t cols):
        cdef int i,j,k
        for i in range(dim):
            heat_map[i] = <unsigned char***>malloc(dim1*sizeof(unsigned char**))
            for j in range(dim1):
                heat_map[i][j] = <uchar**>malloc(rows*sizeof(uchar*))
                for k in range(rows):
                    heat_map[i][j][k] = <uchar*>malloc(cols*sizeof(uchar))
        self.zeros(heat_map, dim, dim1,rows, cols)
    @boundscheck(False)
    @wraparound(False)
    @cdivision(False)
    cdef void zeros(self, uchar****&heat_map, size_t dim, size_t dim1, size_t rows, size_t cols ):
        cdef int i, j, k, n
        for i in range(dim):
            for j in range(dim1):
                for k in range(rows):
                    for n in range(cols):
                         heat_map[i][j][k][n] = 10
                         #print("heat_map", i, heat_map[i][j][k][n])
    @boundscheck(False)
    @wraparound(False)
    @cdivision(False)
    cdef inline int max_elements(self, uchar[:]elements,size_t size):
        cdef int i, elem, index
        elem = elements[0]
        index = 0
        for i in range(1,size):
            if(elem<elements[i]):
                elem = elements[i]
                index = i
        return index

    @boundscheck(False)
    @wraparound(False)
    @cdivision(False)
    cdef inline void set_ones(self, uchar****&heat_maps, int index, int step, int x, int w, int y, int h):
        cdef int i,j
        for i in range(y, h):
            for j in range(x, w):
                heat_maps[index][step][i][j] =255

    @boundscheck(False)
    @wraparound(False)
    @cdivision(False)
    cdef inline pair[Points, Points] bb_dimensions(self, int x, int y, int w, int h, double[:]bb_o, int clip):
        cdef:
            Points limit_1, limit_2
            pair[Points, Points] dimensions

        w = <int>(bb_o[0]*100)
        h = <int>(bb_o[4]*100)

        if (w > clip): w = clip
        if (h > clip): h = clip
        if (w < 1): w = 1
        if (h < 1): h = 1
        x = <int>(((bb_o[2]*50+50)-<int>(w))/2)
        y = <int>(((bb_o[5]*50+50)-<int>(h))/2)

        if (x<0): x = 0
        if (y<0): y = 0
        if (x+w>100): w = 99-x
        if (y+h>100): h = 99-y
        limit_1.start = y
        limit_1.end = y+h
        limit_2.start = x
        limit_2.end = x+w
        #print("start 1 ->", limit_1.start, "end 1 ->", limit_1.end, "start 2->", limit_2.start, "end 2 ->", limit_2.end)
        dimensions.first = limit_1
        dimensions.second = limit_2
        return dimensions

    @boundscheck(False)
    @wraparound(False)
    @cdivision(False)
    cdef void create_heatmaps(self, uchar[:,:,:] labels, double[:,:,:] positions):

        cdef:
            size_t i, j, k
            int x,w,y,h
            float x_c, y_c
            pair[Points, Points] dimensions
            double[:] bb_o
            int clip = 30, lab_t =0, lab_o=0

        for i in range(self.dim2): # ACHTUNG NUR EIN EINZIGES BILD WIRD ITERIERT
            w=0
            h=0
            x=0
            y=0
            if(i>0):
                if(x>self.dim3-1 or x>self.dim4-1 or w>self.dim3-1 or w>self.dim4-1):
                    x_c = 0.0
                else:
                    x_c +=3.8
                if(y>self.dim3-1 or y>self.dim4-1 or h>self.dim3-1 or h>self.dim4-1):
                    y_c= 0.0
                else:
                    y_c += 1.5
            for j in range(self.images):
                if(i<self.dim2):
                    lab_t = self.max_elements(labels[j,i,:], <int>self.dim1)
                    if(lab_t>1):
                        print("index labels ->", lab_t)
                bb_o = positions[j,i,:]
                dimensions = self.bb_dimensions(x,y,w,h,bb_o,clip)
                x = <int>((<float>dimensions.second.start)+x_c)
                w = <int>((<float>dimensions.second.end) + x_c)
                y = <int>((<float>dimensions.first.start)+y_c)
                h = <int>((<float>dimensions.first.end)+y_c)
                if(x>self.dim3-1 or x>self.dim4-1 or w>self.dim3-1 or w>self.dim4-1):
                    x_c = 0.0
                if(y>self.dim3-1 or y>self.dim4-1 or h>self.dim3-1 or h>self.dim4-1):
                    y_c= 0.0
                print(" dimensions ->", lab_t, asarray(labels[j,i]), x, w, y, h, self.dim1, self.dim2, self.dim3, self.dim4, self.images)
                for k in range(self.dim1):
                    if(k>0 and i>0):
                        if(k==1 or k==3):
                            x +=15
                            w +=15
                            y +=16
                            h +=16
                            if (k == lab_t):
                                self.set_ones(self.heat_maps, k, i, <int>x, <int>w, <int>y,<int> h)
                            x -=15
                            w -=15
                            y -=16
                            h -=16
                        else:
                            y+=2
                            h+=2
                            if (k == lab_t):
                                self.set_ones(self.heat_maps, k, i, <int>x, <int>w, <int>y,<int> h)
                    elif (k == lab_t):
                        self.set_ones(self.heat_maps, k, i, <int>x, <int>w, <int>y,<int> h)

    @boundscheck(False)
    @wraparound(False)
    @cdivision(False)
    cdef uchar[:,:,:,:] convert_heatmap(self):
        cdef uchar [:,:,:,:] output = zeros((self.dim1, self.dim2, self.dim3, self.dim4),dtype=uint8)
        cdef int i, j, k, n
        for i in range(self.dim1):
            for j in range(self.dim2):
                for k in range( self.dim3):
                    for n in range(self.dim4):
                        output[i,j,k,n] = self.heat_maps[i][j][k][n]
        return output

    def get_heatmap(self):
        return asarray(self.convert_heatmap(),dtype=uint8)