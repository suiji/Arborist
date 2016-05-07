# distutils: language = c++
# distutils: sources = forest.cc

from libcpp.vector cimport vector

from .cyrowrank cimport RowRank


cdef extern from 'forest.h':
    ctypedef unsigned int UInt # workaround due to cython bug
    cdef cppclass ForestNode:
        void SplitUpdate(RowRank *)
        void Export(vector[unsigned int] &, 
            vector[ForestNode] &,
            vector[vector[UInt]] &,
            vector[vector[UInt]] &,
            vector[vector[double]] &)
        void Init()
        void Set(unsigned int, unsigned int, double)
        unsigned int &Pred()
        double &Num()
        unsigned int &LeafIdx()
        void Ref(unsigned &, unsigned int, double)
