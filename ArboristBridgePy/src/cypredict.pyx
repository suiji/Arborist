from .cypredict cimport Predict


cdef class PyPredict:
    cdef Predict *thisptr