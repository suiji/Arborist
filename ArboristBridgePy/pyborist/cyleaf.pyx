cdef class PyPtrVecBagRow:
    cdef set(self, shared_ptr[vector[BagRow]] ptr):
        self.thisptr = ptr
        return self
    cdef shared_ptr[vector[BagRow]] get(self):
        return self.thisptr
    def __repr__(self):
        return '<Pointer to vector<BagRow>>'



cdef class PyPtrVecLeafNode:
    cdef set(self, shared_ptr[vector[LeafNode]] ptr):
        self.thisptr = ptr
        return self
    cdef shared_ptr[vector[LeafNode]] get(self):
        return self.thisptr
    def __repr__(self):
        return '<Pointer to vector<LeafNode>>'
