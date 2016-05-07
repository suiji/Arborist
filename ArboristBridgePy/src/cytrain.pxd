# distutils: language = c++
# distutils: sources = train.cc

from libcpp.vector cimport vector

from .cyforest cimport ForestNode
from .cyleaf cimport LeafNode
from .cyleaf cimport BagRow


cdef extern from 'train.h':
    cdef cppclass Train:
        Train(vector[unsigned int] &,
            unsigned int,
            vector[double] &,
            vector[unsigned int] &,
            vector[unsigned int] &,
            double,
            vector[ForestNode] &,
            vector[unsigned int] &,
            vector[unsigned int] &,
            vector[LeafNode] &,
            vector[BagRow] &,
            vector[double] &) except +

        Train(vector[double] &,
            vector[unsigned int] &,
            vector[unsigned int] &,
            vector[unsigned int] &, 
            double,
            vector[ForestNode] &,
            vector[unsigned int] &,
            vector[unsigned int] &,
            vector[LeafNode] &,
            vector[BagRow] &,
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
            double, vector[ForestNode] &,
            vector[unsigned int] &, vector[unsigned int] &,
            vector[LeafNode] &, vector[BagRow] &,
            vector[unsigned int] &)

        @staticmethod
        void Classification(int,
            int, int,
            vector[unsigned int] &, int,
            vector[double] &, vector[unsigned int] &,
            vector[unsigned int] &, double,
            vector[ForestNode] &, vector[unsigned int] &,
            vector[unsigned int] &, vector[LeafNode] &,
            vector[BagRow] &, vector[double] &)
