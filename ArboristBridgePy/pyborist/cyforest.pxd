# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared



cdef extern from 'forest.h':
    cdef cppclass ForestNode:
        pass



cdef class PyPtrVecForestNode:
    cdef shared_ptr[vector[ForestNode]] thisptr
    cdef set(self, shared_ptr[vector[ForestNode]] ptr)
    cdef shared_ptr[vector[ForestNode]] get(self)
