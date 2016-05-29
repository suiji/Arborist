cdef class PyRowRank:
    @staticmethod
    def PreSortNum(double[:] _feNum not None,
        unsigned int _nPredNum,
        unsigned int _nRow,
        int[:] _row not None,
        int[:] _rank not None,
        int[:] _invNum not None):
        return RowRank_PreSortNum(&_feNum[0],
            _nPredNum,
            _nRow,
            &_row[0],
            &_rank[0],
            &_invNum[0])

    @staticmethod
    def PreSortFac(int[:] _feFac not None,
        unsigned int _nPredNum,
        unsigned int _nPredFac,
        unsigned int _nRow,
        int[:] _row not None,
        int[:] _rank not None):
        RowRank_PreSortFac(&_feFac[0],
            _nPredNum,
            _nPredFac,
            _nRow,
            &_row[0],
            &_rank[0])
