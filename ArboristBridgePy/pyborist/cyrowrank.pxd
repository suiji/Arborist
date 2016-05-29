# distutils: language = c++



cdef extern from 'rowrank.h':
    cdef cppclass RowRank:
        pass

    cdef void RowRank_PreSortNum 'RowRank::PreSortNum'(const double _feNum[],
            unsigned int _nPredNum,
            unsigned int _nRow,
            int _row[],
            int _rank[],
            int _invNum[])

    cdef void RowRank_PreSortFac 'RowRank::PreSortFac'(const int _feFac[],
            unsigned int _nPredNum,
            unsigned int _nPredFac,
            unsigned int _nRow,
            int _row[],
            int _rank[])
