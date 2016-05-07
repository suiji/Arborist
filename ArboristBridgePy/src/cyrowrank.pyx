from .cyrowrank cimport RowRank


cdef class PyRowRank:
    cdef RowRank *thisptr
