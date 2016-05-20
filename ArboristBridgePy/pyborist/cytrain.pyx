#from libcpp cimport bool # .pxd
#from libcpp.vector cimport vector # .pxd

import numpy as np
cimport numpy as np
from cython cimport view
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr, make_shared

from .cyforest cimport PyForestNode, PyPtrVecForestNode
from .cyleaf cimport PyLeafNode, PyPtrVecLeafNode
from .cyleaf cimport PyBagRow, PyPtrVecBagRow

ctypedef vector[unsigned int] VecUInt # workaround deal to cython bug



def match(a, b, dtype=np.uintc):
    """http://stackoverflow.com/questions/4110059/pythonor-numpy-equivalent-of-match-in-r"""
    a = np.array(a)
    b = np.array(b)
    return np.ascontiguousarray(np.array([np.nonzero(b == x)[0][0] 
        if x in b else None for x in a], dtype=dtype))



cdef class PyTrain:
    @staticmethod
    def Regression(double[::view.contiguous] X not None, #F
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

        print('X');print(np.asarray(X))

        Train_Init(&X[0],
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

        yRanked = np.empty(y.shape[0], dtype=np.double)
        yRanked[:] = y
        yRanked.sort()
        cdef unsigned int[:] row2Rank = match(y, yRanked)
        print('row2Rank');print(np.asarray(row2Rank))

        cdef VecUInt origin = VecUInt(nTree)
        cdef VecUInt facOrig = VecUInt(nTree)
        cdef VecUInt leafOrigin = VecUInt(nTree)
        cdef double[:] predInfo = np.zeros(nPred)

        cdef shared_ptr[vector[ForestNode]] ptrVecForestNode = make_shared[vector[ForestNode]]()
        cdef shared_ptr[vector[LeafNode]] ptrVecLeafNode = make_shared[vector[LeafNode]]()
        cdef shared_ptr[vector[BagRow]] ptrVecBagRow = make_shared[vector[BagRow]]()

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
            deref(ptrVecForestNode),
            facSplit,
            leafOrigin,
            deref(ptrVecLeafNode),
            deref(ptrVecBagRow),
            rank)

        result = {
            'forest': {
                'origin': np.asarray(origin, dtype=np.uintc),
                'facOrig': np.asarray(facOrig, dtype=np.uintc),
                'facSplit': np.asarray(facSplit, dtype=np.uintc),
                'forestNode': PyPtrVecForestNode().set(ptrVecForestNode)
            },
            'leaf': {
                'leafOrigin': np.asarray(leafOrigin, dtype=np.uintc),
                'leafNode': PyPtrVecLeafNode().set(ptrVecLeafNode),
                'bagRow': PyPtrVecBagRow().set(ptrVecBagRow),
                'nRow': nRow, #nRow in train
                'rank': np.asarray(rank, dtype=np.uintc),
                'yRanked': np.asarray(yRanked) # old y getting sorted
            },
            'predInfo': np.asarray(predInfo)
        }
        return result

    @staticmethod
    def Classification():
        pass
