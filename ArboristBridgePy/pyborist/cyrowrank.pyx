import numpy as np
cimport numpy as np

# https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC


cdef class PyRowRank:
    cdef RowRank *thisptr

    def __cinit__(self,
        np.ndarray[int, ndim=1, mode="c"] _feRow not None,
        np.ndarray[int, ndim=1, mode="c"] _feRank not None,
        np.ndarray[int, ndim=1, mode="c"] _feInvNum not None,
        unsigned int _nRow,
        unsigned int _nPredDense):
        self.thisptr = new RowRank(&_feRow[0],
            &_feRank[0],
            &_feInvNum[0],
            _nRow,
            _nPredDense)

    def __dealloc__(self):
        del self.thisptr

    def Lookup(self, predIdx, idx, _rank):
        return self.thisptr.Lookup(predIdx, idx, _rank)

    def Rank2Row(self, predIdx, _rank):
        return self.thisptr.Rank2Row(predIdx, _rank)

    def MeanRank(self, predIdx, rkMean):
        return self.thisptr.MeanRank(predIdx, rkMean)

    @staticmethod
    def PreSortNum(np.ndarray[double, ndim=1, mode="c"] _feNum not None,
        _nPredNum,
        _nRow,
        np.ndarray[int, ndim=1, mode="c"] _row not None,
        np.ndarray[int, ndim=1, mode="c"] _rank not None,
        np.ndarray[int, ndim=1, mode="c"] _invNum not None):
        return RowRank_PreSortNum(&_feNum[0],
            _nPredNum,
            _nRow,
            &_row[0],
            &_rank[0],
            &_invNum[0])

    @staticmethod
    def PreSortFac(np.ndarray[int, ndim=1, mode="c"] _feFac not None,
        _nPredNum,
        _nPredFac,
        _nRow,
        np.ndarray[int, ndim=1, mode="c"] _row not None,
        np.ndarray[int, ndim=1, mode="c"] _rank not None):
        RowRank_PreSortFac(&_feFac[0],
            _nPredNum,
            _nPredFac,
            _nRow,
            &_row[0],
            &_rank[0])
