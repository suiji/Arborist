cdef class PyForestNode:
    def __cinit__(self, bool isRealObj = True):
        if isRealObj:
            self.thisptr = new ForestNode()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    @staticmethod
    cdef factory(ForestNode forestNode):
        #TODO will we have troubles (?wild pointer?) here?
        fn = PyForestNode(isRealObj = False)
        fn.thisptr = &forestNode
        return fn

    def Init(self):
        if self.thisptr:
            return self.thisptr.Init()

    def Set(self,
        unsigned int _pred,
        unsigned int _bump,
        double _num):
        if self.thisptr:
            return self.thisptr.Set(_pred,
                _bump,
                _num)

    def Pred(self):
        if self.thisptr:
            return self.thisptr.Pred()

    def Num(self):
        if self.thisptr:
            return self.thisptr.Num()

    def LeafIdx(self):
        if self.thisptr:
            return self.thisptr.LeafIdx()

    def Nonterminal(self):
        if self.thisptr:
            return self.thisptr.Nonterminal()

    def Ref(self,
        unsigned int &_pred,
        unsigned int &_bump,
        double &_num):
        if self.thisptr:
            return self.thisptr.Ref(_pred, _bump, _num)


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
