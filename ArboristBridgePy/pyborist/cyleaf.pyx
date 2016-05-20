from libcpp cimport bool



cdef class PyBagRow:
    def __init__(self,
        unsigned int row,
        unsigned int sCount):
        self.row = row
        self.sCount = sCount

    def __repr__(self):
        return '<PyBagRow, row={}, sCount={}>'.format(self.row, self.sCount)

    @staticmethod
    cdef wrap(BagRow bagRow):
        return PyBagRow(bagRow.Row(), bagRow.SCount())

    @staticmethod
    cdef BagRow unwrap(PyBagRow pyBagRow):
        cdef BagRow h
        h.Set(pyBagRow.row, pyBagRow.sCount)
        return h

cdef class PyPtrVecBagRow:
    cdef set(self, shared_ptr[vector[BagRow]] ptr):
        self.thisptr = ptr
        return self
    cdef shared_ptr[vector[BagRow]] get(self):
        return self.thisptr
    def __repr__(self):
        return '<Pointer to vector<BagRow>>'




cdef class PyLeafNode:
    def __init__(self, unsigned int extent, double score):
        self.extent = extent
        self.score = score

    def __repr__(self):
        return '<PyLeafNode, extent={}, score={}>'.format(self.extent, self.score)

    @staticmethod
    cdef wrap(LeafNode leafNode):
        return PyLeafNode(leafNode.Extent(), leafNode.GetScore())

    @staticmethod
    cdef LeafNode unwrap(PyLeafNode pyLeafNode):
        cdef LeafNode h
        cdef double *insideScore = &h.Score()
        (&insideScore)[0][0] = pyLeafNode.score
        cdef unsigned int *insideExtent = &h.Count()
        (&insideExtent)[0][0] = pyLeafNode.extent
        return h

cdef class PyPtrVecLeafNode:
    cdef set(self, shared_ptr[vector[LeafNode]] ptr):
        self.thisptr = ptr
        return self
    cdef shared_ptr[vector[LeafNode]] get(self):
        return self.thisptr
    def __repr__(self):
        return '<Pointer to vector<LeafNode>>'
