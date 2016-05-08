# distutils: language = c++

from libcpp cimport bool
from libcpp.vector cimport vector

#TODO
# how to CALL static members in cython?
# declaring using @staticmethod is useless!
# see https://groups.google.com/forum/#!searchin/cython-users/static/cython-users/xaErUq2yY0s/eadFRPSdm0sJ
# see (history of) https://github.com/cython/cython/blob/master/docs/src/userguide/wrapping_CPlusPlus.rst#static-member-method



cdef extern from 'predblock.h':
    cdef cppclass PredBlock:
        pass
    
    cdef unsigned int PredBlock_FacFirst 'PredBlock::FacFirst'()

    cdef bool PredBlock_IsFactor 'PredBlock::IsFactor'(unsigned int predIdx)

    cdef unsigned int PredBlock_BlockIdx 'PredBlock::BlockIdx'(int predIdx, bool &isFactor)

    cdef unsigned int PredBlock_NRow 'PredBlock::NRow'()

    cdef int PredBlock_NPred 'PredBlock::NPred'()

    cdef int PredBlock_NPredFac 'PredBlock::NPredFac'()

    cdef int PredBlock_NPredNum 'PredBlock::NPredNum'()

    cdef int PredBlock_NumFirst 'PredBlock::NumFirst'()

    cdef int PredBlock_NumIdx 'PredBlock::NumIdx'(int predIdx)

    cdef int PredBlock_NumSup 'PredBlock::NumSup'()

    cdef int PredBlock_FacSup 'PredBlock::FacSup'()


    cdef cppclass PBTrain(PredBlock):
        pass

    cdef unsigned int PBTrain_cardMax 'PBTrain::cardMax'

    cdef void PBTrain_Immutables 'PBTrain::Immutables'(double *_feNum,
        int _feCard[],
        const int _cardMax,
        const unsigned int _nPredNum,
        const unsigned int _nPredFac,
        const unsigned int _nRow)

    cdef void PBTrain_DeImmutables 'PBTrain::DeImmutables'()

    cdef double PBTrain_MeanVal 'PBTrain::MeanVal'(int predIdx,
        int rowLow,
        int rowHigh)

    cdef int PBTrain_FacCard 'PBTrain::FacCard'(int predIdx)

    cdef int PBTrain_CardMax 'PBTrain::CardMax'()


    cdef cppclass PBPredict(PredBlock):
        pass

    cdef double *PBPredict_feNumT 'PBPredict::feNumT'

    cdef int *PBPredict_feFacT 'PBPredict::feFacT'

    cdef void PBPredict_Immutables 'PBPredict::Immutables'(double *_feNumT,
        int *_feFacT,
        const unsigned int _nPredNum,
        const unsigned int _nPredFac,
        const unsigned int _nRow)

    cdef void PBPredict_DeImmutables 'PBPredict::DeImmutables'()

    cdef double *PBPredict_RowNum 'PBPredict::RowNum'(unsigned int row)

    cdef int *PBPredict_RowFac 'PBPredict::RowFac'(unsigned int row)
