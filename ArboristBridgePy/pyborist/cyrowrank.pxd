# distutils: language = c++

from libcpp.vector cimport vector


cdef extern from 'rowrank.h':
    cdef cppclass RowRank:
        RowRank(const int _feRow[],
            const int _feRank[],
            const int _feInvNum[],
            unsigned int _nRow,
            unsigned int _nPredDense) except +

        unsigned int Lookup(unsigned int predIdx,
            unsigned int idx,
            unsigned int &_rank)

        unsigned int Rank2Row(unsigned int predIdx,
            int _rank)

        double MeanRank(unsigned int predIdx,
            double rkMean)

        @staticmethod
        void PreSortNum(const double _feNum[],
            unsigned int _nPredNum,
            unsigned int _nRow,
            int _row[],
            int _rank[],
            int _invNum[])
        
        @staticmethod
        void PreSortFac(const int _feFac[],
            unsigned int _nPredNum,
            unsigned int _nPredFac,
            unsigned int _nRow,
            int _row[],
            int _rank[])
