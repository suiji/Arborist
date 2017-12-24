cdef class PyPtrVecBagLeaf:
    cdef set(self, shared_ptr[vector[BagLeaf]] ptr):
        self.thisptr = ptr
        return self
    cdef shared_ptr[vector[BagLeaf]] get(self):
        return self.thisptr
    def __repr__(self):
        return '<Pointer to vector<BagLeaf>>'



cdef class PyPtrVecLeafNode:
    cdef set(self, shared_ptr[vector[LeafNode]] ptr):
        self.thisptr = ptr
        return self
    cdef shared_ptr[vector[LeafNode]] get(self):
        return self.thisptr
    def __repr__(self):
        return '<Pointer to vector<LeafNode>>'
