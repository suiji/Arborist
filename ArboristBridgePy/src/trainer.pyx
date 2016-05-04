# distutils: language = c++
# distutils: sources = train.cc

from libcpp.vector cimport vector


cdef extern from 'train.h':
    cdef cppclass Train:
        Train(vector[unsigned int] &,
            unsigned int,
            vector[double] &,
            vector[unsigned int] &,
            vector[unsigned int] &,
            double,
            vector[class ForestNode] &,
            vector[unsigned int] &,
            vector[unsigned int] &,
            vector[class LeafNode] &,
            vector[class BagRow] &,
            vector[double] &) except +

        Train(vector[double] &,
            vector[unsigned int] &,
            vector[unsigned int] &,
            vector[unsigned int] &, 
            double _predInfo[],
            vector[class ForestNode] &,
            vector[unsigned int] &,
            vector[unsigned int] &,
            vector[class LeafNode] &,
            vector[class BagRow] &,
            vector[unsigned int] &) except +

        @staticmethod
        void Init(double *,
            int, int, int, int, 
            int, int, int, double, 
            bool, int, int, double, 
            int, int, int, double, 
            double)

        @staticmethod
        void Regression(int, 
            int, int,
            vector[double] &, vector[unsigned int] &,
            vector[unsigned int] &, vector[unsigned int] &,
            double, vector[class ForestNode] &,
            vector[unsigned int] &, vector[unsigned int] &,
            vector[class LeafNode] &, vector[class BagRow] &,
            vector[unsigned int] &_rank)

        @staticmethod
        void Classification(int,
            int, int,
            vector[unsigned int] &, int,
            vector[double] &, vector[unsigned int] &,
            vector[unsigned int] &, double _predInfo[],
            vector[class ForestNode] &, vector[unsigned int] &,
            vector[unsigned int] &, vector[class LeafNode] &,
            vector[class BagRow] &, vector[double] &)


cdef class PyTrain:
    cdef Train *thisptr
