# distutils: language = c++
# distutils: sources = forest.cc

from libcpp.vector cimport vector


cdef extern from 'forest.h':
    cdef cppclass ForestNode:
        void SplitUpdate(RowRank *)
        void Export(vector[unsigned int] &, 
            vector[ForestNode] &,
            vector[vector[unsigned int]] &,
            vector[vector[unsigned int]] &,
            vector[vector[double]] &)
        void Init()
        void Set(unsigned int, unsigned int, double)
        unsigned int &Pred()
        double &Num()
        unsigned int &LeafIdx()
        void Ref(unsigned &, unsigned int, double)


cdef class PyForestNode:
    cdef ForestNode *thisptr
