from .cybv cimport BV
from .cybv cimport BitMatrix

cdef class PyBV:
    cdef BV *thisptr

cdef class PyBitMatrix:
    cdef BitMatrix *thisptr
