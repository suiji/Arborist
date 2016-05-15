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
        return [PyForestNode.wrap(x) for x in vec]
    @staticmethod
    cdef LeafNode(vector[LeafNode] vec):
        return [PyLeafNode.wrap(x) for x in vec]
    @staticmethod
    cdef BagRow(vector[BagRow] vec):
        return [PyBagRow.wrap(x) for x in vec]


cdef class PyTrain:
    @staticmethod
    def Regression(double[::view.contiguous, :] X not None, #F
        double[::view.contiguous] y not None,
        unsigned int nRow,
        int nPred,
        int[::view.contiguous] feRow not None,
        int[::view.contiguous] feRank not None,
        int[::view.contiguous] invNum not None,
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
            'predInfo': np.asarray(predInfo)
        }
        return result

    @staticmethod
    def Classification():
        pass
