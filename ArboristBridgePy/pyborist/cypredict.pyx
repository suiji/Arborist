import numpy as np
cimport numpy as np
from cython cimport view
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared

from .cyforest cimport ForestNode, PyPtrVecForestNode
from .cyleaf cimport LeafNode, PyPtrVecLeafNode
from .cyleaf cimport BagLeaf, PyPtrVecBagLeaf



cdef class PyPredict:
    @staticmethod
    def Regression(double[::view.contiguous] valNum not None,
    	unsigned int[::view.contiguous] rowStart not None,
	unsigned int[::view.contiguous] runLength not None,
	unsigned int[::view.contiguous] predStart not None,
	double [::view.contiguous] blockNumT not None,
	unsigned int [::view.contiguous] blockFacT not None,
        unsigned int nPredNum,
        unsigned int nPredFac,
        PyPtrVecForestNode pyPtrForestNode,
        unsigned int [::view.contiguous] origin not None,
	unsigned int nTree,
        unsigned int [::view.contiguous] facSplit not None,
	unsigned int facLen,
	unsigned int [::view.contiguous] facOff not None,
	unsigned int nFac,
	unsigned int [::view.contiguous] leafOrigin,
        PyPtrVecLeafNode pyPtrLeafNode,
	unsigned int [::view.contiguous] bagBits,
	double [::view.contiguous] yTrain not None) :

        cdef unsigned int nRow = np.asarray(yTrain).shape[0]
        cdef unsigned int nLeaf = np.asarray(pyPtrLeafNode).shape[0]
        cdef vector[double] yPred = vector[double](nRow)

        Predict_Regression(np.asarray(valNum),
	    np.asarray(rowStart),
            np.asarray(runLength),
	    np.asarray(predStart),
	    &blockNumT[0],
	    &blockFacT[0],
            nPredNum,
            nPredFac,
            &deref(pyPtrForestNode.get())[0],
            &origin[0],
	    nTree,
            &facSplit[0],
	    facLen,
            &facOff[0],
	    nFac,
            np.asarray(leafOrigin),
            &deref(pyPtrLeafNode.get())[0],
	    nLeaf,
            &bagBits[0],
	    np.asarray(yTrain),
	    yPred)

        return np.asarray(yPred)


    @staticmethod
    def Classification(double[::view.contiguous] valNum not None,
    	unsigned int[::view.contiguous] rowStart not None,
	unsigned int[::view.contiguous] runLength not None,
	unsigned int[::view.contiguous] predStart not None,
	double [::view.contiguous] blockNumT not None,
	unsigned int [::view.contiguous] blockFacT not None,
        unsigned int nPredNum,
        unsigned int nPredFac,
        PyPtrVecForestNode pyPtrForestNode,
        unsigned int [::view.contiguous] origin not None,
	unsigned int nTree,
        unsigned int [::view.contiguous] facSplit not None,
	unsigned int facLen,
	unsigned int [::view.contiguous] facOff not None,
	unsigned int nFac,
	unsigned int [::view.contiguous] leafOrigin,
        PyPtrVecLeafNode pyPtrLeafNode,
	unsigned int [::view.contiguous] bagBits,
        unsigned int rowTrain,
        double[::view.contiguous] weight):

        cdef double[:] probCore = np.zeros(nRow*ctgWidth, dtype=np.double)
        cdef int[:] censusCore = np.empty(nRow*ctgWidth, dtype=np.intc)

        cdef vector[int] yPred = vector[int](nRow)
        cdef vector[unsigned int] yTest # empty
        cdef vector[double] misPredCore # empty

        Predict_Regression(np.asarray(valNum),
	    np.asarray(rowStart),
            np.asarray(runLength),
	    np.asarray(predStart),
	    &blockNumT[0],
	    &blockFacT[0],
            nPredNum,
            nPredFac,
            &deref(pyPtrForestNode.get())[0],
            &origin[0],
	    nTree,
            &facSplit[0],
	    facLen,
            &facOff[0],
	    nFac,
            np.asarray(leafOrigin),
            &deref(pyPtrLeafNode.get())[0],
	    nLeaf,
            &bagBits[0],
	    &rowTrain[0],
	    &weight[0],
	    ctgWidth,
	    np.asarray(yPred),
	    &census[0],
	    np.asarray(yTest),
	    &conft[0],
	    np.asarray(error),
	    &prob[0])


        return (np.asarray(yPred),
            np.asarray(censusCore).reshape(nRow, ctgWidth),
            np.asarray(probCore).reshape(nRow, ctgWidth))
