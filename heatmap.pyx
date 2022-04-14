# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.stdlib cimport malloc,free
from libcpp.string cimport string
from libcpp.utility cimport pair
from numpy import uint8, zeros,asarray,float64, ones
from cython cimport boundscheck, wraparound, cdivision
from cython.parallel import prange, parallel
#from cython_modules.cython_opencvMat.opencv_mat cimport *
ctypedef unsigned char uchar
cdef struct Points:
    int start
    int end

cdef string[6] headers = {b'NHC',b'HC',b'NFC',b'FC',b'NC',b'DPC'}

cdef class HeatMaps:
    cdef:
        string*headers
        uchar **** heat_maps
        int last_loc, rows, loc_seq
        # Mat**heat_maps

    def __cinit__(self, uchar[:,:,:,:,:] features, uchar[:,:,:] labels, double[:,:,:] positions):

        self.last_loc = 1
        self.rows = labels.shape[2] # 6
        self.loc_seq = positions.shape[1] # 11
        self.headers = headers
        self.heatmaps(labels, positions)
        if(self.heat_maps==NULL):
            raise MemoryError()

    def __deallocate__(self):
        #free(self.headers)
        free(self.heat_maps)
    """
    cdef inline void populate_mat(self, Mat**&heat_maps, size_t rows, size_t cols):
        cdef int i
        heat_maps = <Mat**>malloc(cols*sizeof(Mat*))
        for i in range(rows):
            heat_maps[i] = <Mat*>malloc(cols*sizeof(Mat))
    cdef void populate_labels(self, string*&outlabels, string[6] inlabels):
        cdef int i
        for i in range(6):
            outlabels[i] = inlabels[i]

    cdef void printstrings(self, string*out, size_t size):
        cdef int i
        for i in range(size):
            print(out[i])
    """
    @boundscheck(True)
    @wraparound(True)
    @cdivision(True)
    cdef void populate_heatmaps(self, uchar****&heat_map, size_t rows, size_t cols):
        cdef int i,j,k
        for i in range(rows):
            heat_map[i] = <unsigned char***>malloc(cols*sizeof(unsigned char**))
            for j in range(cols):
                heat_map[i][j] = <uchar**>malloc(120*sizeof(uchar*))
                for k in range(120):
                    heat_map[i][j][k] = <uchar*>malloc(160*sizeof(uchar))
    @boundscheck(True)
    @wraparound(True)
    cdef void zeros(self, uchar****&heat_map, size_t rows, size_t cols ):
        cdef int i, j, k, n
        for i in range(rows):
            for j in range(cols):
                for k in range(120):
                    for n in range(160):
                        heat_map[i][j][k][n] = 0
                        #print("heat_map", i, heat_map[i][j][k][n])
    @boundscheck(True)
    @wraparound(True)
    @cdivision(True)
    cdef inline void set_ones(self, int index, int step, Points limit_1, Points limit_2):
        cdef int i,j

        for i in range(limit_1.start, limit_1.end):
            for j in range(limit_2.start, limit_2.end):
                self.heat_maps[index][step][i][j] +=1
                #print("set ->", self.heat_maps[index][step][i][j])

    @boundscheck(True)
    @wraparound(True)
    @cdivision(True)
    cdef inline int max_elements(self, uchar[:]elements,size_t size):
        cdef int i, elem, index
        elem = elements[0]
        index = 0
        for i in range(1,size):
            if(elem<elements[i]):
                elem = elements[i]
                index = i
        return index

    @boundscheck(True)
    @wraparound(True)
    @cdivision(True)
    cdef inline double* np2pointer(self, double[:] position, size_t rows):
        cdef:
            int i
            double * pos =<double*>malloc(rows*sizeof(double))
        assert(rows==6), "the number of rows in np array is not equal to the pointer array "
        #print("location ->", asarray(position, dtype=float64))
        if not (pos):
            raise MemoryError()
        try:

            for i in range(rows):
                pos[i] = position[i]
            return pos
        finally:
            free(pos)
    cdef pair[Points, Points] bb_dimensions(self, int x, int y, int w, int h, double[:]bb_o, int clip):
        cdef:
            Points limit_1, limit_2
            pair[Points, Points] dimensions

        w = <int>(bb_o[0]*100)
        h = <int>(bb_o[4]*100)
        #print("w ->", w, "h ->", h)
        if (w > clip): w = clip
        if (h > clip): h = clip
        if (w < 1): w = 1
        if (h < 1): h = 1
        x = <int>(((bb_o[2]*50+50)-<int>(w))/2)
        y = <int>(((bb_o[5]*50+50)-<int>(h))/2)
        #print("x ->", x, "y ->", y, "w ->", w, "h ->", h)
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

    @boundscheck(True)
    @wraparound(True)
    @cdivision(True)
    cdef void heatmaps(self, uchar[:,:,:] labels, double[:,:,:] positions):

        cdef:
            int i, j, k, lab_t =0, lab_o=0, w=0, h=0, x=0, y=0, count=0
            #double* bb_o
            double[:] bb_o
            Points limit_1, limit_2
            pair[Points, Points] dimensions
            int clip = 30
            int size = positions.shape[0] # 100 if edram else X.shape[0]//2#
            int steps =  labels.shape[1] # 10
            bint predicted = True
        #self.populate_mat(self.heap_maps, 6,10)
        self.heat_maps = <unsigned char****>malloc(self.rows*sizeof(unsigned char***))
        self.populate_heatmaps(self.heat_maps,self.rows,self.loc_seq)
        self.zeros(self.heat_maps, self.rows, self.loc_seq)
        for i in range(steps+self.last_loc): # ACHTUNG NUR EIN EINZIGES BILD WIRD ITERIERT
            for j in range(size):
                if(i<steps):
                    lab_t = self.max_elements(labels[j,i,:], self.rows)
                    if(lab_t>1):
                       print("index labels ->", lab_t)
                       count +=1
                #bb_o = self.np2pointer(positions[i,j,:], positions.shape[2])
                bb_o = positions[j,i,:]
                #print("labels array ->", asarray(labels[j,i,:], dtype=uint8))
                dimensions = self.bb_dimensions(x,y,w,h, bb_o, clip)

                for k in range(self.rows):
                    if (k == lab_t):
                        #self.heatmap[e,step,y:y+h,x:x+w] = self.heat_map[e,step,y:y+h,x:x+w] + 1
                        self.set_ones(k, i, dimensions.first, dimensions.second) #limit_1, limit_2)
            print("count -> ", count)
    @boundscheck(True)
    @wraparound(True)
    @cdivision(True)
    cdef uchar[:,:,:,:] convert_heatmap(self, int rows, int cols):
        cdef uchar [:,:,:,:] output = ones((rows,cols,120,160),dtype=uint8)
        cdef int i, j, k, n
        for i in range(rows):
            for j in range(cols):
                for k in range(120):
                    for n in range(160):
                        # if(self.heat_maps[i][j][k][n]>0):
                            # print("keys ->",i,j,k,n,"value ->",self.heat_maps[i][j][k][n])
                        output[i,j,k,n] = self.heat_maps[i][j][k][n]
        return output

    def get_heatmap(self):
        return asarray(self.convert_heatmap(6,11),dtype=uint8)