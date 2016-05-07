from .cyforest cimport ForestNode


cdef class PyForestNode:
    cdef ForestNode *thisptr
