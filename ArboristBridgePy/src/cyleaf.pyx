from .cyleaf cimport BagRow
from .cyleaf cimport LeafNode


cdef class PyBagRow:
    cdef BagRow *thisptr


cdef class PyLeadNode:
    cdef LeafNode *thisptr


