# distutils: language = c++
# distutils: sources = forest.cc

from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from 'bv.h':
    ctypedef unsigned int UInt # workaround due to cython bug


    cdef cppclass BV:
        BV(unsigned int, 
            bool) except + # len -> _leng because python
        
        unsigned int NBit()

        void Consume(vector[unsigned int] &out,
            unsigned int bitEnd)

        BV *Resize(unsigned int bitMin)

        unsigned int Slots()

        @staticmethod
        unsigned int SlotBits()

        @staticmethod
        unsigned int SlotAlign(unsigned int) # len -> _leng because python

        @staticmethod
        unsigned int SlotMask(unsigned int pos,
            unsigned int &mask)

        bool Test(unsigned int slot,
            unsigned int mask)

        bool IsSet(unsigned int pos)

        void SetBit(unsigned int pos)

        unsigned int Slot(unsigned int slot)

        void SetSlot(unsigned int slot,
            unsigned int val)


    cdef cppclass BitMatrix(BV):
        BitMatrix(unsigned int _nRow,
            unsigned int _nCol) except +

        BitMatrix(unsigned int _nRow,
            unsigned int _nCol,
            const vector[unsigned int] &_raw) except +

        void SetColumn(const BV *vec,
            int colIdx)

        @staticmethod
        void Export(const vector[unsigned int] &_raw,
            unsigned int _nRow,
            vector[vector[UInt]] &vecOut)

        bool IsSet(unsigned int row,
            unsigned int col)

        void SetBit(unsigned int row,
            unsigned int col)
