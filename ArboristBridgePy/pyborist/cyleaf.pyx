from libcpp cimport bool



cdef class PyBagRow:
    def __cinit__(self, bool isRealObj = True):
        if isRealObj:
            self.thisptr = new BagRow()
    
    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    @staticmethod
    cdef factory(BagRow bagRow):
        self = PyBagRow(isRealObj = False)
        self.thisptr = &bagRow
        return self

    def Init(self):
        if self.thisptr:
            return self.thisptr.Init()

    def Set(self,
        unsigned int _row,
        unsigned int _sCount):
        if self.thisptr:
            return self.thisptr.Set(_row,
                _sCount)

    def Row(self):
        if self.thisptr:
            return self.thisptr.Row()

    def SCount(self):
        if self.thisptr:
            return self.thisptr.SCount()

    def Ref(self,
        _row,
        _sCount):
        if self.thisptr:
            return self.thisptr.Ref(_row,
                _sCount)



cdef class PyLeafNode:
    def __cinit__(self, bool isRealObj = True):
        if isRealObj:
            self.thisptr = new LeafNode()
    
    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    @staticmethod
    cdef factory(LeafNode leafNode):
        self = PyLeafNode(isRealObj = False)
        self.thisptr = &leafNode
        return self

    def Init(self):
        if self.thisptr:
            return self.thisptr.Init()

    def Extent(self):
        if self.thisptr:
            return self.thisptr.Extent()

    def Count(self):
        if self.thisptr:
            return self.thisptr.Count()

    def Score(self):
        if self.thisptr:
            return self.thisptr.Score()

    def GetScore(self):
        if self.thisptr:
            return self.thisptr.GetScore()
