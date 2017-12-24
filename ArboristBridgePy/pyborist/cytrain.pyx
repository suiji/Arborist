#from libcpp cimport bool # .pxd
#from libcpp.vector cimport vector # .pxd

from libcpp cimport nullptr

import numpy as np
cimport numpy as np
from cython cimport view
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr, make_shared

from .cyforest cimport ForestNode, PyPtrVecForestNode
from .cyleaf cimport LeafNode, PyPtrVecLeafNode
from .cyleaf cimport BagLeaf, PyPtrVecBagLeaf

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
        unsigned int nPred,
        int[::view.contiguous] feRow not None,
        int[::view.contiguous] feRank not None,
        int[::view.contiguous] invNum not None,
        unsigned int nTree,
        unsigned int nSamp,
        double[::view.contiguous] sampleWeight not None,
        bool withRepl,
	bool thinLeaves,
        unsigned int trainBlock,
        unsigned int minNode,
        double minRatio,
        unsigned int totLevels,
	unsigned int leafMax,
        unsigned int predFixed,
	double[::view.contiguous] splitQuant not None,
        double[::view.contiguous] predProb not None,
        double[::view.contiguous] regMono not None):

        Train_Init(nPred,
		nTree,
		nSamp,
		np.asarray(sampleWeight),
		withRepl,
		trainBlock,
		minNode,
		minRatio,
		totLevels,
		leafMax,
		0, #cardMax
		predFixed,
		&splitQuant[0],
		&predProb[0],
		thinLeaves,
		&regMono[0])
#		&X[0],

        yRanked = np.empty(y.shape[0], dtype=np.double)
        yRanked[:] = y
        yRanked.sort()
        cdef unsigned int[:] row2Rank = match(y, yRanked)

        cdef VecUInt origin = VecUInt(nTree)
        cdef VecUInt facOrig = VecUInt(nTree)
        cdef VecUInt leafOrigin = VecUInt(nTree)
        cdef double[:] predInfo = np.zeros(nPred)

        cdef shared_ptr[vector[ForestNode]] ptrVecForestNode = make_shared[vector[ForestNode]]()
        cdef shared_ptr[vector[LeafNode]] ptrVecLeafNode = make_shared[vector[LeafNode]]()
        cdef shared_ptr[vector[BagLeaf]] ptrVecBagLeaf = make_shared[vector[BagLeaf]]()

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
            deref(ptrVecBagLeaf),
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
                'bagLeaf': PyPtrVecBagLeaf().set(ptrVecBagLeaf),
                'nRow': nRow, #nRow in train
                'rank': np.asarray(rank, dtype=np.uintc),
                'yRanked': np.asarray(yRanked) # old y getting sorted
            },
            'predInfo': np.asarray(predInfo)
        }
        return result


    @staticmethod
    def Classification(double[::view.contiguous] X not None, #F
        unsigned int[::view.contiguous] y not None,
        unsigned int nRow,
        unsigned int nPred,
        int[::view.contiguous] feRow not None,
        int[::view.contiguous] feRank not None,
        int[::view.contiguous] invNum not None,
        unsigned int nTree,
        unsigned int nSamp,
        double[::view.contiguous] sampleWeight not None,
        bool withRepl,
	bool thinLeaves,
        unsigned int trainBlock,
        unsigned int minNode,
        double minRatio,
        unsigned int totLevels,
	unsigned int leafMax,
        unsigned int predFixed,
	double[::view.contiguous] splitQuant not None,
        double[::view.contiguous] predProb not None,
        double[::view.contiguous] classWeightJittered not None):

        cdef unsigned int ctgWidth = np.max(y) + 1 # how many categories

        Train_Init(nPred,
		nTree,
		nSamp,
		np.asarray(sampleWeight),
		withRepl,
		trainBlock,
		minNode,
		minRatio,
		totLevels,
		leafMax,
		0, # ctgWidth
		predFixed,
		&splitQuant[0],
		&predProb[0],
		thinLeaves,
		<double *> nullptr)
#		&X[0],

        cdef VecUInt origin = VecUInt(nTree)
        cdef VecUInt facOrig = VecUInt(nTree)
        cdef VecUInt leafOrigin = VecUInt(nTree)
        cdef double[:] predInfo = np.zeros(nPred, dtype=np.double)

        cdef shared_ptr[vector[ForestNode]] ptrVecForestNode = make_shared[vector[ForestNode]]()
        cdef shared_ptr[vector[LeafNode]] ptrVecLeafNode = make_shared[vector[LeafNode]]()
        cdef shared_ptr[vector[BagLeaf]] ptrVecBagLeaf = make_shared[vector[BagLeaf]]()

        cdef VecUInt facSplit
        cdef vector[double] weight

        Train_Classification(&feRow[0],
            &feRank[0],
            &invNum[0],
            np.asarray(y),
            ctgWidth,
            np.asarray(classWeightJittered),
            origin,
            facOrig,
            &predInfo[0],
            deref(ptrVecForestNode),
            facSplit,
            leafOrigin,
            deref(ptrVecLeafNode),
            deref(ptrVecBagLeaf),
            weight)

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
                'bagLeaf': PyPtrVecBagLeaf().set(ptrVecBagLeaf),
                'nRow': nRow, #nRow in train
                'weight': np.asarray(weight, dtype=np.double),
                'yLevels': np.unique(y) # old y all different levels
            },
            'predInfo': np.asarray(predInfo)
        }
        return result
