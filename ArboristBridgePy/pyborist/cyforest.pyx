cdef class PyPtrVecForestNode:
    cdef set(self, shared_ptr[vector[ForestNode]] ptr):
        self.thisptr = ptr
        return self
    cdef shared_ptr[vector[ForestNode]] get(self):
        return self.thisptr
    def __repr__(self):
        return '<Pointer to vector<ForestNode>>'
