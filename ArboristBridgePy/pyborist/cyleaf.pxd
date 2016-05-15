# distutils: language = c++

from libcpp.vector cimport vector



cdef extern from 'leaf.h':
    ctypedef unsigned int UInt # workaround due to cython bug


    cdef cppclass BagRow:
        void Init()

        void Set(unsigned int _row,
            unsigned int _sCount)

        unsigned int Row()

        unsigned int SCount()

        void Ref(unsigned int &_row,
            unsigned int &_sCount)


    cdef cppclass LeafNode:
        void Init()

        unsigned int Extent()

        unsigned int &Count()

        double &Score()

        double GetScore()

    cdef void LeafNode_Export 'LeafNode::Export'(const vector[unsigned int] &_origin,
        const vector[LeafNode] &_leafNode,
        vector[vector[double]] &_score,
        vector[vector[UInt]] &_extent)
        
    cdef unsigned int LeafNode_LeafCount 'LeafNode::LeafCount'(const vector[unsigned int] &_origin,
        unsigned int height,
        unsigned int tIdx)



cdef class PyBagRow:
    cdef public unsigned int row
    cdef public unsigned int sCount
    @staticmethod
    cdef wrap(BagRow bagRow)
    @staticmethod
    cdef BagRow unwrap(PyBagRow pyBagRow)




cdef class PyLeafNode:
    cdef public unsigned int extent
    cdef public double score
    @staticmethod
    cdef wrap(LeafNode leafNode)
    @staticmethod
    cdef LeafNode unwrap(PyLeafNode pyLeafNode)
