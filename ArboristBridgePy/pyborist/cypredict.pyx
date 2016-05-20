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
        unsigned int[::view.contiguous] origin not None,
        unsigned int[::view.contiguous] facOrig not None,
        unsigned int[::view.contiguous] facSplit not None,
        PyPtrVecForestNode pyPtrForestNode,
        double[::view.contiguous] yRanked,
        unsigned int[::view.contiguous] leafOrigin,
        PyPtrVecLeafNode pyPtrLeafNode,
        PyPtrVecBagRow pyPtrBagRow,
        unsigned int rowTrain,
        unsigned int[::view.contiguous] rank
        ):
        cdef double[:] yPred = np.empty(nRow, dtype=np.double)

        Predict_Regression(&X[0],
            NULL, #blockFacT
            nPred,
            0, #nPredFac
            deref(pyPtrForestNode.get()),
            np.asarray(origin),
            np.asarray(facOrig),
            np.asarray(facSplit),
            np.asarray(leafOrigin),
            deref(pyPtrLeafNode.get()),
            deref(pyPtrBagRow.get()),
            np.asarray(rank),
            np.asarray(yRanked),
            np.asarray(yPred),
            0)

        return np.asarray(yPred)
