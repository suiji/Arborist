import numpy as np
cimport numpy as np
from cython cimport view
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr, make_shared

from .cyforest cimport PyForestNode, ForestNode, PyPtrVecForestNode
from .cyleaf cimport PyLeafNode, LeafNode, PyPtrVecLeafNode
from .cyleaf cimport PyBagRow, BagRow, PyPtrVecBagRow



cdef class PyPredict:
    @staticmethod
    def Regression(double[::view.contiguous] X not None, #C
        unsigned int nRow,
        unsigned int nPred,
        unsigned int[::view.contiguous] _origin not None,
        unsigned int[::view.contiguous] _facOrig not None,
        unsigned int[::view.contiguous] _facSplit not None,
        PyPtrVecForestNode pyPtrForestNode,
        double[::view.contiguous] _yRanked,
        unsigned int[::view.contiguous] _leafOrigin,
        PyPtrVecLeafNode pyPtrLeafNode,
        PyPtrVecBagRow pyPtrBagRow,
        unsigned int rowTrain,
        unsigned int[::view.contiguous] _rank
        ):
        cdef vector[unsigned int] origin = np.asarray(_origin)
        cdef vector[unsigned int] facOrig = np.asarray(_facOrig)
        cdef vector[unsigned int] facSplit = np.asarray(_facSplit)
        cdef vector[double] yRanked = np.asarray(_yRanked)
        cdef vector[unsigned int] leafOrigin = np.asarray(_leafOrigin)
        cdef vector[unsigned int] rank = np.asarray(_rank)

        cdef vector[double] yPred = vector[double](nRow)

        Predict_Regression(&X[0],
            NULL, #blockFacT
            nPred,
            0, #nPredFac
            deref(pyPtrForestNode.get()),
            origin,
            facOrig,
            facSplit,
            leafOrigin,
            deref(pyPtrLeafNode.get()),
            deref(pyPtrBagRow.get()),
            rank,
            yRanked,
            yPred,
            0)

        return np.asarray(yPred)
