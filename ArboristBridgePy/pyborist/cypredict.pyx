import numpy as np
cimport numpy as np
from cython cimport view
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared

from .cyforest cimport ForestNode, PyPtrVecForestNode
from .cyleaf cimport LeafNode, PyPtrVecLeafNode
from .cyleaf cimport BagRow, PyPtrVecBagRow



cdef class PyPredict:
    @staticmethod
    def Regression(double[::view.contiguous] X not None,
        unsigned int nRow,
        unsigned int nPred,
        unsigned int[::view.contiguous] origin not None,
        unsigned int[::view.contiguous] facOrig not None,
        unsigned int[::view.contiguous] facSplit not None,
        PyPtrVecForestNode pyPtrForestNode,
        double[::view.contiguous] yRanked,
        unsigned int[::view.contiguous] leafOrigin,
        PyPtrVecLeafNode pyPtrLeafNode,
        PyPtrVecBagRow pyPtrBagRow,
        unsigned int rowTrain,
        unsigned int[::view.contiguous] rank):
        cdef vector[double] yPred = vector[double](nRow)

        Predict_Regression(&X[0],
            NULL, # blockFacT
            nPred,
            0, # nPredFac
            deref(pyPtrForestNode.get()),
            np.asarray(origin),
            np.asarray(facOrig),
            np.asarray(facSplit),
            np.asarray(leafOrigin),
            deref(pyPtrLeafNode.get()),
            deref(pyPtrBagRow.get()),
            np.asarray(rank),
            np.asarray(yRanked),
            yPred,
            0)

        return np.asarray(yPred)


    @staticmethod
    def Classification(double[::view.contiguous] X not None,
        unsigned int nRow,
        unsigned int nPred,
        unsigned int ctgWidth,
        unsigned int[::view.contiguous] origin not None,
        unsigned int[::view.contiguous] facOrig not None,
        unsigned int[::view.contiguous] facSplit not None,
        PyPtrVecForestNode pyPtrForestNode,
        unsigned int[::view.contiguous] yLevels,
        unsigned int[::view.contiguous] leafOrigin,
        PyPtrVecLeafNode pyPtrLeafNode,
        PyPtrVecBagRow pyPtrBagRow,
        unsigned int rowTrain,
        double[::view.contiguous] weight):

        cdef double[:] probCore = np.zeros(nRow*ctgWidth, dtype=np.double)
        cdef int[:] censusCore = np.empty(nRow*ctgWidth, dtype=np.intc)

        cdef vector[int] yPred = vector[int](nRow)
        cdef vector[unsigned int] yTest # empty
        cdef vector[double] misPredCore # empty

        Predict_Classification(&X[0],
            NULL, # blockFacT
            nPred,
            0, # nPredFac
            deref(pyPtrForestNode.get()),
            np.asarray(origin),
            np.asarray(facOrig),
            np.asarray(facSplit),
            np.asarray(leafOrigin),
            deref(pyPtrLeafNode.get()),
            deref(pyPtrBagRow.get()),
            np.asarray(weight),
            yPred,
            &censusCore[0],
            yTest,
            NULL, #_conf,
            misPredCore,
            &probCore[0],
            0)

        return (np.asarray(yPred),
            np.asarray(censusCore).reshape(nRow, ctgWidth),
            np.asarray(probCore).reshape(nRow, ctgWidth))
