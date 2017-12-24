# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared



cdef extern from 'leaf.h':
    cdef cppclass BagLeaf:
        pass

    cdef cppclass LeafNode:
        pass



cdef class PyPtrVecBagLeaf:
    cdef shared_ptr[vector[BagLeaf]] thisptr
    cdef set(self, shared_ptr[vector[BagLeaf]] ptr)
    cdef shared_ptr[vector[BagLeaf]] get(self)



cdef class PyPtrVecLeafNode:
    cdef shared_ptr[vector[LeafNode]] thisptr
    cdef set(self, shared_ptr[vector[LeafNode]] ptr)
    cdef shared_ptr[vector[LeafNode]] get(self)
