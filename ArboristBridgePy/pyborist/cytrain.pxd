# distutils: language = c++

from libcpp cimport bool
from libcpp.vector cimport vector

from .cyforest cimport ForestNode
from .cyleaf cimport LeafNode
from .cyleaf cimport BagLeaf



cdef extern from 'train.h':
    cdef void Train_Init 'Train::Init'(unsigned int _nPred,
       unsigned int _nTree,
       unsigned int _nSamp,
       const vector[double] &_feSampleWeight,
       bool withRepl,
       unsigned int _trainBlock,
       unsigned int _minNode,
       double _minRatio,
       unsigned int _totLevels,
       unsigned int _leafMax,
       unsigned int _ctgWidth,
       unsigned int _predFixed,
       const double _splitQuant[],
       const double _predProb[],
       bool _thinLeaves,
       const double _regMono[])

    cdef void Train_Regression 'Train::Regression'(int _feRow[],
        int _feRank[],
        int _feInvNum[],
        const vector[double] &_y,
        const vector[unsigned int] &_row2Rank,
        vector[unsigned int] &_origin,
        vector[unsigned int] &_facOrigin,
        double _predInfo[],
        vector[ForestNode] &_forestNode,
        vector[unsigned int] &_facSplit,
        vector[unsigned int] &_leafOrigin,
        vector[LeafNode] &_leafNode,
        vector[BagLeaf] &_bagLeaf,
        vector[unsigned int] &_rank)

    cdef void Train_Classification 'Train::Classification'(int _feRow[],
        int _feRank[],
        int _feInvNum[],
        const vector[unsigned int]  &_yCtg,
        int _ctgWidth,
        const vector[double] &_yProxy,
        vector[unsigned int] &_origin,
        vector[unsigned int] &_facOrigin,
        double _predInfo[],
        vector[ForestNode] &_forestNode,
        vector[unsigned int] &_facSplit,
        vector[unsigned int] &_leafOrigin,
        vector[LeafNode] &_leafNode,
        vector[BagLeaf] &_bagLeaf,
        vector[double] &_weight)
