# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared



cdef extern from 'leaf.h':
    cdef cppclass BagRow:
        pass

    cdef cppclass LeafNode:
        pass



cdef class PyPtrVecBagRow:
    cdef shared_ptr[vector[BagRow]] thisptr
    cdef set(self, shared_ptr[vector[BagRow]] ptr)
    cdef shared_ptr[vector[BagRow]] get(self)



cdef class PyPtrVecLeafNode:
    cdef shared_ptr[vector[LeafNode]] thisptr
    cdef set(self, shared_ptr[vector[LeafNode]] ptr)
    cdef shared_ptr[vector[LeafNode]] get(self)
