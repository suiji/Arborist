# distutils: language = c++

from libcpp.vector cimport vector

from .cyforest cimport Forest


cdef extern from 'pretree.h':
    cdef cppclass PreTree:
        PreTree(unsigned int _bagCount) except +

        @staticmethod
        void Immutables(unsigned int _nPred,
            unsigned int _nSamp,
            unsigned int _minH)

        @staticmethod
        void DeImmutables()

        @staticmethod
        void Reserve(unsigned int height)

        vector[unsigned int] DecTree(Forest *forest,
            unsigned int tIdx,
            double predInfo[])

        void NodeConsume(Forest *forest,
            unsigned int tIdx)

        unsigned int BitWidth()

        void BitConsume(unsigned int *outBits)
