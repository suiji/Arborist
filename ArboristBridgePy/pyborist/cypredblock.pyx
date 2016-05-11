import numpy as np
cimport numpy as np

#TODO how to return pointers then convert them into numpy arrays...


cdef class PyPredBlock:
    @staticmethod
    def FacFirst():
        return PredBlock_FacFirst()

    @staticmethod
    def IsFactor(predIdx):
        return PredBlock_IsFactor(predIdx)

    @staticmethod
    def BlockIdx(predIdx, isFactor):
        return PredBlock_BlockIdx(predIdx, isFactor)

    @staticmethod
    def NRow():
        return PredBlock_NRow()

    @staticmethod
    def NPred():
        return PredBlock_NPred()

    @staticmethod
    def NPredFac():
        return PredBlock_NPredFac()

    @staticmethod
    def NPredNum():
        return PredBlock_NPredNum()

    @staticmethod
    def NumFirst():
        return PredBlock_NumFirst()

    @staticmethod
    def NumIdx(predIdx):
        return PredBlock_NumIdx(predIdx)

    @staticmethod
    def NumSup():
        return PredBlock_NumSup()

    @staticmethod
    def FacSup():
        return PredBlock_FacSup()



cdef class PyPBTrain(PyPredBlock):
    @staticmethod
    def Immutables(np.ndarray[double, ndim=1, mode="c"] _feNum not None,
        np.ndarray[int, ndim=1, mode="c"] _feCard not None,
        _cardMax,
        _nPredNum,
        _nPredFac,
        _nRow):
        return PBTrain_Immutables(&_feNum[0],
            &_feCard[0],
            _cardMax,
            _nPredNum,
            _nPredFac,
            _nRow)

    @staticmethod
    def DeImmutables():
        return PBTrain_DeImmutables()

    @staticmethod
    def MeanVal(predIdx,
        rowLow,
        rowHigh):
        return PBTrain_MeanVal(predIdx,
            rowLow,
            rowHigh)

    @staticmethod
    def FacCard(predIdx):
        return PBTrain_FacCard(predIdx)

    @staticmethod
    def CardMax():
        return PBTrain_CardMax()



cdef class PyPBPredict(PyPredBlock):
    @staticmethod
    def Immutables(np.ndarray[double, ndim=1, mode="c"] _feNumT not None,
        np.ndarray[int, ndim=1, mode="c"] _feFacT not None,
        _nPredNum,
        _nPredFac,
        _nRow):
        return PBPredict_Immutables(&_feNumT[0],
            &_feFacT[0],
            _nPredNum,
            _nPredFac,
            _nRow)

    @staticmethod
    def DeImmutables():
        return PBPredict_DeImmutables()

    #@staticmethod
    #def feNumT():
    #    return PBPredict_feNumT()

    #@staticmethod
    #def feFacT():
    #    return PBPredict_feFacT()

    #@staticmethod
    #def RowNum(row):
    #    return PBPredict_RowNum(row)

    #@staticmethod
    #def RowFac(row):
    #    return PBPredict_RowFac(row)
