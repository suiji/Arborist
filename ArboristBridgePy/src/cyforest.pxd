# distutils: language = c++

from libcpp cimport bool
from libcpp.vector cimport vector

from .cybv cimport BitMatrix
from .cybv cimport BV
from .cypretree cimport PreTree
from .cyrowrank cimport RowRank
from .cypredict cimport Predict


cdef extern from 'forest.h':
    ctypedef unsigned int UInt # workaround due to cython bug


    cdef cppclass ForestNode:
        void SplitUpdate(const RowRank *rowRank)

        void Export(const vector[unsigned int] &_nodeOrigin,
            const vector[ForestNode] &_forestNode,
            vector[vector[UInt]] &_pred,
            vector[vector[UInt]] &_bump,
            vector[vector[double]] &_split)

        void Init()

        void Set(unsigned int _pred,
            unsigned int _bump,
            double _num)

        unsigned int &Pred()

        double &Num()

        unsigned int &LeafIdx()

        bool Nonterminal()

        void Ref(unsigned int &_pred,
            unsigned int &_bump,
            double &_num)


    cdef cppclass Forest:
        Forest(vector[ForestNode] &_forestNode,
            vector[unsigned int] &_origin,
            vector[unsigned int] &_facOrigin,
            vector[unsigned int] &_facVec) except +

        Forest(vector[ForestNode] &_forestNode,
            vector[unsigned int] &_origin,
            vector[unsigned int] &_facOrigin,
            vector[unsigned int] &_facVec,
            Predict *_predict) except +

        void SplitUpdate(const RowRank *rowRank)

        void PredictAcross(unsigned int rowStart,
            unsigned int rowEnd,
            const BitMatrix *bag)

        void PredictRowNum(unsigned int row,
            const double rowT[],
            unsigned int rowBlock,
            const BitMatrix *bag)

        void PredictRowFac(unsigned int row,
            const int rowT[],
            unsigned int rowBlock,
            const BitMatrix *bag)

        void PredictRowMixed(unsigned int row,
            const double rowNT[],
            const int rowIT[],
            unsigned int rowBlock,
            const BitMatrix *bag)

        void NodeInit(unsigned int treeHeight)

        int NTree()

        unsigned int *Origin()

        void TreeBlock(PreTree *ptBlock[],
            int treeBlock,
            int treeStart)

        unsigned int Origin(int tIdx)

        unsigned int NodeIdx(unsigned int tIdx,
            unsigned int nodeOffset)

        void NonterminalProduce(unsigned int tIdx,
            unsigned int nodeIdx,
            unsigned int _predIdx,
            unsigned int _bump,
            double _split) 

        void LeafProduce(unsigned int tIdx,
            unsigned int nodeIdx,
            unsigned int _leafIdx) 

        void Reserve(unsigned int nodeEst,
            unsigned int facEst,
            double slop)

        unsigned int Height()

        unsigned int TreeHeight(int tIdx)

        unsigned int SplitHeight()

        bool Nonterminal(unsigned int idx)

        bool Nonterminal(int tIdx,
            unsigned int off)

        unsigned int &LeafIdx(unsigned int idx)

        unsigned int &LeafIdx(int tIdx,
            unsigned int off)

        void NodeProduce(unsigned int _predIdx,
            unsigned int _bump,
            double _split)

        void BitProduce(const BV *splitBits,
            unsigned int bitEnd)

        void Origins(unsigned int tIdx)
