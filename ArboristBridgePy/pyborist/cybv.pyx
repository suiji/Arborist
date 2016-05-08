# distutils: language = c++

cdef class PyBV:
    cdef BV *thisptr
    def __cinit__(self, leng, slotWise=False):
        self.thisptr = new BV(leng, slotWise)
    def __dealloc__(self):
        del self.thisptr

cdef class PyBitMatrix:
    cdef BitMatrix *thisptr

cdef class PyBVJagged:
    cdef BVJagged *thisptr
