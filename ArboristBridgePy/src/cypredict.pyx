# distutils: language = c++
# distutils: sources = predict.cc

from libcpp.vector cimport vector


cdef extern from 'predict.h':
    cdef cppclass Predict:
        Predict(int, unsigned int, unsigned int) except +

        @staticmethod
        void Regression(double *,
            int *, unsigned int,
            unsigned int, vector[ForestNode] &, 
            vector[unsigned int] &, vector[unsigned int] &,
            vector[unsigned int] &, vector[unsigned int] &,
            vector[LeafNode] &, vector[BagRow] &,
            vector[unsigned int] &, vector[double] &,
            vector[double] &, unsigned int)

        @staticmethod
        void Quantiles(double *,
            int *, unsigned int,
            unsigned int, vector[ForestNode] &,
            vector[unsigned int] &, vector[unsigned int] &,
            vector[unsigned int] &, vector[unsigned int] &,
            vector[LeafNode] &, vector[BagRow] &,
            vector[unsigned int] &, vector[double] &,
            vector[double] &, vector[double] &,
            unsigned int, vector[double] &,
            unsigned int bagTrain)

        @staticmethod
        void Classification(double *,
            int *, unsigned int,
            unsigned int, vector[ForestNode] &,
            vector[unsigned int] &, vector[unsigned int] &,
            vector[unsigned int] &, vector[unsigned int] &,
            vector[LeafNode] &, vector[BagRow] &,
            vector[double] &, vector[int] &,
            int *, vector[unsigned int] &,
            int *, vector[double] &,
            double *, unsigned int)


cdef class PyPredict:
    cdef Predict *thisptr