from .cyforest cimport ForestNode
from .cyforest cimport Forest

cdef class PyForestNode:
    cdef ForestNode *thisptr

cdef class PyForest:
    cdef Forest *thisptr
