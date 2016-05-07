# distutils: language = c++

from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from 'predblock.h':
    cdef cppclass PredBlock:
        @staticmethod
        unsigned int FacFirst()

        @staticmethod
        bool IsFactor(unsigned int predIdx)

        @staticmethod
        unsigned int BlockIdx(int predIdx, bool &isFactor)

        @staticmethod
        unsigned int NRow()

        @staticmethod
        int NPred()

        @staticmethod
        int NPredFac()

        @staticmethod
        int NPredNum()

        @staticmethod
        int NumFirst()

        @staticmethod
        int NumIdx(int predIdx)

        @staticmethod
        int NumSup()

        @staticmethod
        int FacSup()


    cdef cppclass PBTrain(PredBlock):
        #TODO how to deal with static FIELD?!
        #@staticmethod
        #unsigned int cardMax

        @staticmethod
        void Immutables(double *_feNum,
            int _feCard[],
            const int _cardMax,
            const unsigned int _nPredNum,
            const unsigned int _nPredFac,
            const unsigned int _nRow)

        @staticmethod
        void DeImmutables()

        @staticmethod
        double MeanVal(int predIdx,
            int rowLow,
            int rowHigh)

        @staticmethod
        int FacCard(int predIdx)

        @staticmethod
        int CardMax()


    cdef cppclass PBPredict(PredBlock):
        #TODO how to deal with static FIELD?!
        #@staticmethod
        #double *feNumT

        #TODO how to deal with static FIELD?!
        #@staticmethod
        #int *feFacT

        @staticmethod
        void Immutables(double *_feNumT,
            int *_feFacT,
            const unsigned int _nPredNum,
            const unsigned int _nPredFac,
            const unsigned int _nRow)

        @staticmethod
        void DeImmutables()

        @staticmethod
        double *RowNum(unsigned int row)

        @staticmethod
        int *RowFac(unsigned int row)
