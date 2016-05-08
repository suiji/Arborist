cdef class PyForestNode:
    cdef ForestNode *thisptr

cdef class PyForest:
    cdef Forest *thisptr
