cdef class PyForestNode:
    def __init__(self,
        unsigned int pred,
        unsigned int bump,
        double num):
        self.pred = pred
        self.bump = bump
        self.num = num

    def __repr__(self):
        return ('<PyForestNode, pred={}, bump={}, num={}>'.format(
            self.pred, self.bump, self.num))

    @staticmethod
    cdef wrap(ForestNode forestNode):
        cdef unsigned int pred = 0
        cdef unsigned int bump = 0
        cdef double num = 0.0
        forestNode.Ref(pred, bump, num)
        return PyForestNode(pred, bump, num)

    @staticmethod
    cdef ForestNode unwrap(PyForestNode pyForestNode):
        cdef ForestNode h
        h.Set(pyForestNode.pred,
            pyForestNode.bump,
            pyForestNode.num)
        return h

cdef class PyPtrVecForestNode:
    cdef set(self, shared_ptr[vector[ForestNode]] ptr):
        self.thisptr = ptr
        return self
    cdef shared_ptr[vector[ForestNode]] get(self):
        return self.thisptr
    def __repr__(self):
        return '<Pointer to vector<ForestNode>>'



cdef class PyForest:
    cdef Forest *thisptr

    #def __cinit__(self,
    #    vector[ForestNode] &_forestNode,
    #    vector[unsigned int] &_origin,
    #    vector[unsigned int] &_facOrigin,
    #    vector[unsigned int] &_facVec):
    #    self.thisptr = new Forest(_forestNode,
    #        _origin,
    #        _facOrigin,
    #        _facVec)

    #def __dealloc__(self):
    #    del self.thisptr
