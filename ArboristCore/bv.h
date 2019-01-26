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

#ifndef ARBORIST_BV_H
#define ARBORIST_BV_H

#include <vector>
#include <algorithm>

#include "typeparam.h"

// TODO: Reparametrize with templates.

class BV {
  const unsigned int nSlot; // Number of typed (uint) slots.
  unsigned int *raw;
  const bool wrapper;  // True iff an overlay onto pre-allocated memory.
  
 public:
  static const unsigned int full = 1;
  static const unsigned int eltSize = 1;
  static const unsigned int slotSize = sizeof(unsigned int);
  static constexpr unsigned int slotElts = 8 * slotSize;

  BV(size_t len, bool slotWise = false);
  BV(const vector<unsigned int> &raw_);
  BV(unsigned int raw_[], size_t nSlot_);
  BV(vector<unsigned int> &raw_, unsigned int nSlot_);

  ~BV();

  inline void Serialize(unsigned char *bbRaw) const {
    for (size_t i = 0; i < nSlot * sizeof(unsigned int); i++) {
      bbRaw[i] = *((unsigned char *) &raw[0] + i);
    }
  }

  /**
     @brief Accessor for position within the 'raw' buffer.
   */
  inline unsigned int *Raw(unsigned int off) {
    return raw + off;
  }

  /**
     @brief Appends contents onto output vector.

     @param[out] out outputs the raw bit vector contents.

     @param bitEnd specifies a known end position if positive, otherwise
     indicates that a default value be used.

     @return void, with output vector parameter.
  */
  void consume(vector<unsigned int> &out, unsigned int bitEnd = 0) const;

  
  unsigned int PopCount() const;

  BV *Resize(size_t bitMin);

  /**
     @brief Accessor for slot count.
   */
  unsigned int Slots() const {
    return nSlot;
  }
  
  /**
     @brief Accessor for slotwise bit count.

     @return count of bits per slot.
  */
  static unsigned int SlotElts() {
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
  static inline unsigned int SlotAlign(size_t len) {
    return (len + slotElts - 1) / slotElts;
  }


  static inline size_t strideBytes(size_t len) {
    return SlotAlign(len) * sizeof(unsigned int);
  }

  /**
     @return length of aligned row in bits.
   */
  static inline unsigned int Stride(size_t len) {
    return slotElts * SlotAlign(len);
  }

  
  /**
     @brief Builds a bit mask having a single bit high.

     @param pos is a bit position

     @param mask outputs a slot-width mask with the bit at 'pos' set.

     @return slot containing position.
   */
  static inline unsigned int SlotMask(unsigned int pos, unsigned int &mask) {
    unsigned int slot = pos / slotElts;
    mask = full << (pos - (slot * slotElts));

    return slot;
  }


  bool Test(unsigned int slot, unsigned int mask) const {
    return (raw[slot] & mask) == mask;
  }

  
  /**
     @brief Tests the bit at a specified position.

     @param bv is the bit vector implementation.

     @param pos is the bit position to test.

     @return true iff bit position is set in the bit vector.
   */
  inline bool testBit(unsigned int pos) const {
    unsigned int mask;
    unsigned int slot = SlotMask(pos, mask);

    return Test(slot, mask);
  }

  
  /**
     @brief Sets the bit at position 'pos'.

     @param bv is the bit vector implementation.

     @param pos is the position to set.

     @return void.
   */
  inline void setBit(unsigned int pos, bool on = true) {
    unsigned int mask;
    unsigned int slot = SlotMask(pos, mask);
    unsigned int val = raw[slot];
    raw[slot] = on ? (val | mask) : (val & ~mask);
  }


  inline unsigned int Slot(unsigned int slot) const {
    return raw[slot];
  }
  
  
  inline void setSlot(unsigned int slot, unsigned int val) {
    raw[slot] = val;
  }


  inline void Clear() {
    for (unsigned int i = 0; i < nSlot; i++) {
      raw[i] = 0;
    }
  }
};


/**
   @brief Like a bit vector, but with row-major strided access.

 */
class BitMatrix : public BV {
  const unsigned int nRow;
  const unsigned int stride; // Number of uint cells per row.
  void dump(unsigned int _nRow, vector<vector<unsigned int> > &bmOut) const;
  void colDump(unsigned int _nRow,
               vector<unsigned int> &outCol,
               unsigned int colIdx) const;

 public:
  BitMatrix(unsigned int _nRow, unsigned int _nCol);
  BitMatrix(unsigned int _nRow, unsigned int _nCol, const vector<unsigned int> &raw_);
  BitMatrix(unsigned int raw_[], size_t _nRow, size_t _nCol);
  ~BitMatrix();

  inline unsigned int getNRow() const {
    return nRow;
  }


  inline size_t getStride() const {
    return stride;
  }
  

  static void dump(const vector<unsigned int> &raw_,
                   unsigned int _nRow,
                   vector<vector<unsigned int> > &vecOut);


  /**
     @brief Wraps a row section as a bit vector.

     @param row is the row number being accessed.

     @return wrapped bit vector.
   */
  inline shared_ptr<BV> BVRow(unsigned int row) {
    return make_shared<BV>(Raw((row * stride) / slotElts), stride);
  }


  /**
     @brief Bit test with short-circuit for zero-length matrix.

     @return whether bit at specified coordinate is set.
   */
  inline bool testBit(unsigned int row, unsigned int col) const {
    return stride == 0 ? false : BV::testBit(row * stride + col);
  }

  
  inline void setBit(unsigned int row, unsigned int col, bool on = true) {
    BV::setBit(row * stride + col, on);
  }


  inline void clearBit(unsigned int row, unsigned int col) {
    setBit(row, col, false);
  }
};


/**
   @brief Jagged bit matrix:  unstrided access.
 */
class BVJagged : public BV {
  const unsigned int *rowExtent;
  const unsigned int nRow;
  vector<unsigned int> rowDump(unsigned int rowIdx) const;

 public:
  BVJagged(unsigned int raw_[],
           const unsigned int height_[], // Cumulative extent per row.
           unsigned int nRow_);
  ~BVJagged();
  void dump(vector<vector<unsigned int> > &outVec);


  /**
     @brief Bit test for jagged matrix.

     @param row is the (nonstrided) row.

     @param pos is the bit position within the row.

     @return true iff bit set.

   */
  inline bool testBit(unsigned int row, unsigned int pos) const {
    unsigned int mask;
    unsigned int slot = SlotMask(pos, mask);
    unsigned int base = row == 0 ? 0 : rowExtent[row-1];
    
    return Test(base + slot, mask);
  }
};


class CharV {
  unsigned int *raw;
  const unsigned int nSlot;
  const bool wrapper;
 public:
  static const unsigned char full = 0xff;
  static const unsigned int eltSize = 8 * sizeof(unsigned char);
  static const unsigned int slotSize =  8 * sizeof(unsigned int);
  static constexpr unsigned int slotElts = slotSize / eltSize;

  CharV(unsigned int _nSlot);
  CharV(unsigned int *raw_, unsigned int _nSlot);
  ~CharV();

  
  /**
     @brief Accessor for slot count.
   */
  unsigned int Slots() const {
    return nSlot;
  }
  
  /**
     @brief Accessor for slotwise bit count.

     @return count of bits per slot.
  */
  static unsigned int SlotElts() {
    return slotElts;
  }

  
  inline unsigned int *Raw(unsigned int off) {
    return raw + off;
  }

  
  /**
     @brief Aligns length to smallest enclosing buffer size.

     @param len is the element count to align.

     @return number of buffer slots in aligned row.
   */
  static inline unsigned int SlotAlign(unsigned int len) {
    return (len + slotElts - 1) / slotElts;
  }

  static inline unsigned int Stride(unsigned int len) {
    return slotElts * SlotAlign(len);
  }


  inline unsigned char Get(unsigned int pos) const {
    unsigned int slot = pos / slotElts;
    unsigned int slotPos = pos - slot * slotElts;
    unsigned int shiftBits = slotPos * eltSize;
    unsigned int mask = full << shiftBits;
    return (raw[slot] & mask) >> shiftBits;
  }


  /**
     @brief sets slot at position to value passed.

     @param pos is the position to set.

     @return void, with side-effected raw value.
   */
  inline void set(unsigned int pos, unsigned char val) {
    unsigned int slot = pos / slotElts;
    unsigned int slotPos = pos - (slot * slotElts);
    unsigned int shiftBits = slotPos * eltSize;
    unsigned int mask = ~(full << shiftBits);
    unsigned int gappedVal = raw[slot] & mask;
    raw[slot] = gappedVal | (val << shiftBits);
  }

  /**
     @brief Masks off all bits beyond value passed.

     @return masked value.
   */
  inline unsigned char Mask(unsigned int pos, unsigned int del) const {
    unsigned char val = Get(pos);
    return val & ~(0xff << del);
  }
};


/**
 */
class CharRow : public CharV {
 public:
  CharRow(unsigned int *raw_, unsigned int _nSlot) : CharV(raw_, _nSlot) {}
  ~CharRow() {}
};



/**
 */
class CharMatrix : public CharV {
  unsigned int stride;
  unsigned int nCol;
 public:
  CharMatrix(unsigned int nRow, unsigned int nCol);
  inline CharRow *Row(unsigned int row) {
    return new CharRow(Raw((stride * row)/slotElts), stride);
  }
};

#endif
