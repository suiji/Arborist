# distutils: language = c++
# distutils: sources = rowrank.cc

from libcpp.vector cimport vector


cdef extern from 'rowrank.h':
    cdef cppclass RowRank:
        RowRank(int, int,
            int, unsigned int,
            unsigned int) except +
        unsigned int Lookup(unsigned int, unsigned int, unsigned int &)
        unsigned int Rank2Row(unsigned int, int)
        double MeanRank(unsigned int, double)
        @staticmethod
        void PreSortNum(double, unsigned int,
            unsigned int, int,
            int, int)
        @staticmethod
        void PreSortFac(int, unsigned int,
            unsigned int, unsigned int,
            int, int)


cdef class PyRowRank:
    cdef RowRank *thisptr
