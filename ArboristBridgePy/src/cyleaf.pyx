# distutils: language = c++
# distutils: sources = leaf.cc

from libcpp.vector cimport vector


cdef extern from 'leaf.h':
    cdef cppclass BagRow:
        void Init()
        void Set(unsigned int, unsigned int)
        unsigned int SCount()
        void Ref(unsigned int &, unsigned int &)

    cdef cppclass LeafNode:
        void Init()
        unsigned int Extent()
        unsigned int &Count()
        double &Score()
        double GetScore()
        @staticmethod
        unsigned int LeafCount(vector[unsigned int] &, unsigned int, unsigned int)


cdef class PyBagRow:
    cdef BagRow *thisptr


cdef class PyLeadNode:
    cdef LeafNode *thisptr
