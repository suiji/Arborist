// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bv.h

   @brief 1-, 2- and 4-bit packed vectors with integer alignment.

   @author Mark Seligman

 */

#ifndef CORE_BV_H
#define CORE_BV_H

#include <vector>
#include <algorithm>
#include <stdexcept>


#include "typeparam.h"

// TODO: Reparametrize with templates.

typedef unsigned int RawT;

class BV {
  const size_t nSlot; // Number of typed (uint) slots.
  RawT* raw;
  const bool wrapper;  // True iff an overlay onto pre-allocated memory.
  
 public:
  static constexpr unsigned int full = 1;
  static constexpr unsigned int eltSize = 1;
  static constexpr size_t slotSize = sizeof(RawT);
  static constexpr size_t slotElts = 8 * slotSize;

  BV(size_t len, bool slotWise = false);
  BV(const vector<RawT> &raw_);
  BV(RawT raw_[], size_t nSlot_);
  BV(vector<RawT>& raw_, size_t nSlot_);

  ~BV();

  inline void Serialize(unsigned char *bbRaw) const {
    for (size_t i = 0; i < nSlot * sizeof(RawT); i++) {
      bbRaw[i] = *((unsigned char *) &raw[0] + i);
    }
  }


  /**
     @brief Determines whether container is empty.
   */
  bool isEmpty() const {
    return nSlot == 0;
  }

  
  /**
     @brief Accessor for position within the 'raw' buffer.
   */
  inline RawT* Raw(size_t off) {
    return raw + off;
  }

  /**
     @brief Appends contents onto output vector.

     @param[out] out outputs the raw bit vector contents.

     @param bitEnd specifies a known end position if positive, otherwise
     indicates that a default value be used.

     @return void, with output vector parameter.
  */
  void consume(vector<RawT>& out, size_t bitEnd = 0) const;


  BV operator|(const BV& bvR) {
    /*
    if (bvR.getNSlot() != nSlot) {
      throw std::invalid_argument("mismatched bit vector | operation");
      }*/
    BV bvOr(nSlot, true);
    for (size_t i = 0; i < nSlot; i++) {
      bvOr.raw[i] = raw[i] | bvR.raw[i];
    }
    return bvOr;
  }

  
  BV& operator&=(const BV& bvR) {
    /*
    if (nSlot != bvR.getNSlot()) {
      throw std::invalid_argument("mismatched bit vector &= operation");
      }*/

    for (size_t i = 0; i < nSlot; i++) {
      raw[i] &= bvR.raw[i];
    }
    return *this;
  }


  BV& operator|=(const BV& bvR) {
    /*
    if (nSlot != bvR.getNSlot()) {
      throw std::invalid_argument("mismatched bit vector |= operation");
      }*/

    for (size_t i = 0; i < nSlot; i++) {
      raw[i] |= bvR.raw[i];
    }
    return *this;
  }


  BV operator~() {
    BV bvTilde(nSlot, true);
    for (size_t i = 0; i < nSlot; i++) {
      bvTilde.raw[i] = ~raw[i];
    }
    return bvTilde;
  }

  
  BV *Resize(size_t bitMin);

  /**
     @brief Accessor for slot count.
   */
  size_t getNSlot() const {
    return nSlot;
  }
  
  /**
     @brief Accessor for slotwise bit count.

     @return count of bits per slot.
  */
  static size_t getSlotElts() {
    return slotElts;
  }
  
  // Compiler should be generating shifts.  In c++11 can replace
  // with constant expression for log2(slotElts) and introduce
  // shifts explicitly.

  /**
     @brief Aligns element count to the smallest enclosing buffer size.

     @param len is the element count to align.

     @return length of containing aligned quantity in buffer units.
   */
  static inline size_t slotAlign(size_t len) {
    return (len + slotElts - 1) / slotElts;
  }


  static inline size_t strideBytes(size_t len) {
    return slotAlign(len) * sizeof(RawT);
  }

  /**
     @return length of aligned row in bits.
   */
  static inline size_t Stride(size_t len) {
    return slotElts * slotAlign(len);
  }

  
  /**
     @brief Builds a bit mask having a single bit high.

     @param pos is a bit position

     @param mask outputs a slot-width mask with the bit at 'pos' set.

     @return slot containing position.
   */
  static inline size_t slotMask(size_t pos, RawT& mask) {
    size_t slot = pos / slotElts;
    mask = full << (pos - (slot * slotElts));

    return slot;
  }


  bool test(size_t slot, RawT mask) const {
    return (raw[slot] & mask) == mask;
  }

  
  /**
     @brief Tests the bit at a specified position.

     @param bv is the bit vector implementation.

     @param pos is the bit position to test.

     @return true iff bit position is set in the bit vector.
   */
  inline bool testBit(size_t pos) const {
    RawT mask;
    size_t slot = slotMask(pos, mask);

    return test(slot, mask);
  }

  
  /**
     @brief Sets the bit at position 'pos'.

     @param bv is the bit vector implementation.

     @param pos is the position to set.

     @return void.
   */
  inline void setBit(size_t pos, bool on = true) {
    RawT mask;
    size_t slot = slotMask(pos, mask);
    RawT val = raw[slot];
    raw[slot] = on ? (val | mask) : (val & ~mask);
  }


  inline auto Slot(size_t slot) const {
    return raw[slot];
  }
  
  
  inline void setSlot(size_t slot, RawT val) {
    raw[slot] = val;
  }

  /**
     @brief Sets all slots to zero.
   */
  inline void clear() {
    for (size_t i = 0; i < nSlot; i++) {
      raw[i] = 0;
    }
  }


  /**
     @brief Sets all slots high.
   */
  inline void saturate() {
    for (size_t i = 0; i < nSlot; i++) {
      raw[i] = 0xffffffff;
    }
  }
};


/**
   @brief Like a bit vector, but with row-major strided access.

 */
class BitMatrix : public BV {
  const size_t nRow;
  const unsigned int stride; // Number of uint cells per row.
  void dump(size_t _nRow, vector<vector<unsigned int> > &bmOut) const;
  void colDump(size_t _nRow,
               vector<unsigned int> &outCol,
               unsigned int colIdx) const;

 public:
  BitMatrix(size_t _nRow, unsigned int _nCol);
  BitMatrix(size_t _nRow, unsigned int _nCol, const vector<RawT> &raw_);
  BitMatrix(RawT raw_[], size_t _nRow, size_t _nCol);
  ~BitMatrix();

  inline size_t getNRow() const {
    return nRow;
  }


  inline size_t getStride() const {
    return stride;
  }
  

  static void dump(const vector<RawT> &raw_,
                   size_t _nRow,
                   vector<vector<RawT> > &vecOut);


  /**
     @brief Wraps a row section as a bit vector.

     @param row is the row number being accessed.

     @return wrapped bit vector.
   */
  inline shared_ptr<BV> BVRow(size_t row) {
    return make_shared<BV>(Raw((row * stride) / slotElts), stride);
  }


  /**
     @brief Bit test with short-circuit for zero-length matrix.

     @return whether bit at specified coordinate is set.
   */
  inline bool testBit(size_t row, unsigned int col) const {
    return stride == 0 ? false : BV::testBit(row * stride + col);
  }

  
  inline void setBit(size_t row, unsigned int col, bool on = true) {
    BV::setBit(row * stride + col, on);
  }


  inline void clearBit(size_t row, unsigned int col) {
    setBit(row, col, false);
  }
};


/**
   @brief Jagged bit matrix:  unstrided access.
 */
class BVJagged : public BV {
  const unsigned int *rowExtent;
  const size_t nRow;
  vector<unsigned int> rowDump(size_t rowIdx) const;

 public:
  BVJagged(RawT raw_[],
           const unsigned int height_[], // Cumulative extent per row.
           size_t nRow_);
  ~BVJagged();
  void dump(vector<vector<unsigned int> > &outVec);


  /**
     @brief Bit test for jagged matrix.

     @param row is the (nonstrided) row.

     @param pos is the bit position within the row.

     @return true iff bit set.

   */
  inline bool testBit(size_t row, size_t pos) const {
    RawT mask;
    size_t slot = slotMask(pos, mask);
    unsigned int base = row == 0 ? 0 : rowExtent[row-1];
    
    return test(base + slot, mask);
  }
};

#endif
