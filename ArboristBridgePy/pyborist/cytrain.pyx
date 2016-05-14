#from libcpp cimport bool # .pxd
#from libcpp.vector cimport vector # .pxd

import numpy as np
cimport numpy as np
from cython cimport view

from .cyforest cimport PyForestNode
from .cyleaf cimport PyLeafNode
from .cyleaf cimport PyBagRow

ctypedef vector[unsigned int] VecUInt # workaround deal to cython bug



def match(a, b):
    """http://stackoverflow.com/questions/4110059/pythonor-numpy-equivalent-of-match-in-r"""
    a = np.array(a)
    b = np.array(b)
    return np.array([np.nonzero(b == x)[0][0] 
        if x in b else None for x in a], dtype=np.uint)



cdef class PyVecWrapped:
    """turn the cpp vector[T] to python list(PyT)
    """
    @staticmethod
    cdef ForestNode(vector[ForestNode] vec):
        return [PyForestNode.factory(x) for x in vec]
    @staticmethod
    cdef LeafNode(vector[LeafNode] vec):
        return [PyLeafNode.factory(x) for x in vec]
    @staticmethod
    cdef BagRow(vector[BagRow] vec):
        return [PyBagRow.factory(x) for x in vec]



def train_regression(double[::view.contiguous, :] X not None, # Fortran contiguous 2-D matrix?
    double[::view.contiguous] y not None,
    unsigned int nRow,
    int nPred,
    int[::view.contiguous] feRow not None, #rowRank
    int[::view.contiguous] feRank not None, #rowRank
    int[::view.contiguous] invNum not None, #rowRank
    int nTree,
    int nSamp,
    double[::view.contiguous] sampleWeight not None,
    bool withRepl,
    int trainBlock,
    int minNode,
    double minRatio,
    int totLevels,
    int predFixed,
    double[::view.contiguous] predProb not None,
    double[::view.contiguous] regMono not None):

    if nPred <= 0:
        raise ValueError('No predictors here.')

    Train_Init(&X[0][0],
        NULL, #feFacCard,
        0, #cardMax,
        nPred,
        0, #nPredFac,
        nRow,
        nTree,
        nSamp,
        &sampleWeight[0],
        withRepl,
        trainBlock,
        minNode,
        minRatio,
        totLevels,
        0,
        predFixed,
        &predProb[0],
        &regMono[0])

    yRanked = np.empty(y.shape[0])
    yRanked[:] = y
    yRanked.sort()
    cdef unsigned int[:] row2Rank = match(y, yRanked)

    cdef VecUInt origin = VecUInt(nTree)
    cdef VecUInt facOrig = VecUInt(nTree)
    cdef VecUInt leafOrigin = VecUInt(nTree)
    cdef double[:] predInfo = np.zeros(nPred) # maybe .empty()

    cdef vector[ForestNode] forestNode
    cdef vector[LeafNode] leafNode
    cdef vector[BagRow] bagRow
    cdef VecUInt rank
    cdef VecUInt facSplit

    print('sampleWeight  ');print(np.asarray(sampleWeight))
    print('predProb  ');print(np.asarray(predProb))
    print('regMono  ');print(np.asarray(regMono))
    print('ferow  ');print(np.asarray(feRow))
    print('ferank  ');print(np.asarray(feRank))
    print('invNum  ');print(np.asarray(invNum))
    print('y  ');print(np.asarray(y))
    print('row2Rank  ');print(np.asarray(row2Rank))
    print('origin  ');print(np.asarray(origin))
    print('facOrig  ');print(np.asarray(facOrig))
    print('predInfo  ');print(np.asarray(predInfo))
    #forestNode
    print('facSplit  ');print(np.asarray(facSplit))
    print('leafOrigin  ');print(np.asarray(leafOrigin))
    #leafNode
    #bagRow
    print('rank  ');print(np.asarray(rank))



    Train_Regression(&feRow[0],
        &feRank[0],
        &invNum[0],
        np.asarray(y),
        np.asarray(row2Rank),
        origin,
        facOrig,
        &predInfo[0],
        forestNode,
        facSplit,
        leafOrigin,
        leafNode,
        bagRow,
        rank)

    print('finish!')

    result = {
        'forest': [
            origin,
            facOrig,
            facSplit,
            PyVecWrapped.ForestNode(forestNode)
        ],
        'leaf': [
            leafOrigin,
            PyVecWrapped.LeafNode(leafNode),
            PyVecWrapped.BagRow(bagRow),
            nRow,
            rank,
            yRanked
        ],
        'predInfo': predInfo
    }
    return result

#cdef class PyTrain:
#    @staticmethod
#    def Init(np.ndarray[double, ndim=1, mode="c"] _feNum not None,
#        np.ndarray[int, ndim=1, mode="c"]  _facCard not None,
#        _cardMax,
#        _nPredNum,
#        _nPredFac,
#        _nRow,
#        _nTree,
#        _nSamp,
#        np.ndarray[double, ndim=1, mode="c"] _feSampleWeight not None,
#        withRepl,
#        _trainBlock,
#        _minNode,
#        _minRatio,
#        _totLevels,
#        _ctgWidth,
#        _predFixed,
#        np.ndarray[double, ndim=1, mode="c"] _predProb not None,
#        np.ndarray[double, ndim=1, mode="c"] _regMono not None):
#        return Train_Init(&_feNum[0],
#            &_facCard[0],
#            _cardMax,
#            _nPredNum,
#            _nPredFac,
#            _nRow,
#            _nTree,
#            _nSamp,
#            &_feSampleWeight[0],
#            withRepl,
#            _trainBlock,
#            _minNode,
#            _minRatio,
#            _totLevels,
#            _ctgWidth,
#            _predFixed,
#            &_predProb[0],
#            &_regMono[0])

#    @staticmethod
#    def Regression(np.ndarray[int, ndim=1, mode="c"] _feRow not None,
#        np.ndarray[int, ndim=1, mode="c"]  _feRank not None,
#        np.ndarray[int, ndim=1, mode="c"]  _feInvNum not None,
#        _y,
#        _row2Rank,
#        _origin,
#        _facOrigin,
#        np.ndarray[double, ndim=1, mode="c"] _predInfo not None,
#        vector[ForestNode] &_forestNode,
#        _facSplit,
#        _leafOrigin,
#        vector[LeafNode] &_leafNode,
#        vector[BagRow] &_bagRow,
#        _rank):
#        return Train_Regression(&_feRow[0],
#            &_feRank[0],
#            &_feInvNum[0],
#            _y,
#            _row2Rank,
#            _origin,
#            _facOrigin,
#            &_predInfo[0],
#            vector[ForestNode] &_forestNode,
#            _facSplit,
#            _leafOrigin,
#            vector[LeafNode] &_leafNode,
#            vector[BagRow] &_bagRow,
#            _rank)

#    @staticmethod
#    def Classification(np.ndarray[int, ndim=1, mode="c"]  _feRow not None,
#        np.ndarray[int, ndim=1, mode="c"]  _feRank not None,
#        np.ndarray[int, ndim=1, mode="c"]  _feInvNum not None,
#        _yCtg,
#        _ctgWidth,
#        _yProxy,
#        _origin,
#        _facOrigin,
#        np.ndarray[double, ndim=1, mode="c"] _predInfo not None,
#        vector[ForestNode] &_forestNode,
#        _facSplit,
#        _leafOrigin,
#        vector[LeafNode] &_leafNode,
#        vector[BagRow] &_bagRow,
#        _weight):
#        return Train_Classification(&_feRow[0],
#            &_feRank[0],
#            &_feInvNum[0],
#            _yCtg,
#            _ctgWidth,
#            _yProxy,
#            _origin,
#            _facOrigin,
#            &_predInfo[0],
#            vector[ForestNode] &_forestNode,
#            _facSplit,
#            _leafOrigin,
#            vector[LeafNode] &_leafNode,
#            vector[BagRow] &_bagRow,
#            _weight)
