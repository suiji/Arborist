# distutils: language = c++

from libcpp.vector cimport vector

from .cyforest cimport ForestNode
from .cyleaf cimport LeafNode
from .cyleaf cimport BagRow


cdef extern from 'predict.h':
    cdef cppclass Predict:
        Predict(int _nTree,
            unsigned int _nRow,
            unsigned int _nonLeafIdx) except +

        @staticmethod
        void Regression(double *_blockNumT,
            int *_blockFacT,
            unsigned int _nPredNum,
            unsigned int _nPredFac,
            vector[ForestNode] &_forestNode,
            vector[unsigned int] &_origin,
            vector[unsigned int] &_facOff,
            vector[unsigned int] &_facSplit,
            vector[unsigned int] &_leafOrigin,
            vector[LeafNode] &_leafNode,
            vector[BagRow] &_bagRow,
            vector[unsigned int] &_rank,
            const vector[double] &yRanked,
            vector[double] &yPred,
            unsigned int bagTrain)

        @staticmethod
        void Quantiles(double *_blockNumT,
            int *_blockFacT,
            unsigned int _nPredNum,
            unsigned int _nPredFac,
            vector[ForestNode] &_forestNode,
            vector[unsigned int] &_origin,
            vector[unsigned int] &_facOff,
            vector[unsigned int] &_facSplit,
            vector[unsigned int] &_leafOrigin,
            vector[LeafNode] &_leafNode,
            vector[BagRow] &_bagRow,
            vector[unsigned int] &_rank,
            const vector[double] &yRanked,
            vector[double] &yPred,
            const vector[double] &quantVec,
            unsigned int qBin,
            vector[double] &qPred,
            unsigned int bagTrain)

        @staticmethod
        void Classification(double *_blockNumT,
            int *_blockFacT,
            unsigned int _nPredNum,
            unsigned int _nPredFac,
            vector[ForestNode] &_forestNode,
            vector[unsigned int] &_origin,
            vector[unsigned int] &_facOff,
            vector[unsigned int] &_facSplit,
            vector[unsigned int] &_leafOrigin,
            vector[LeafNode] &_leafNode,
            vector[BagRow] &_bagRow,
            vector[double] &_leafInfoCtg,
            vector[int] &yPred,
            int *_census,
            const vector[unsigned int] &_yTest,
            int *_conf,
            vector[double] &_error,
            double *_prob,
            unsigned int bagTrain)
