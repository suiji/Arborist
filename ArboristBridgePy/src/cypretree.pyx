from .cypretree cimport PreTree


cdef class PyPreTree:
    cdef PreTree *thisptr
