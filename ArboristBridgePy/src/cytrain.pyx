from .cytrain cimport Train


cdef class PyTrain:
    cdef Train *thisptr
