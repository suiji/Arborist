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

// TODO: Recast using templates.

class BV {
  const unsigned int nSlot;
  unsigned int *raw;
  const bool wrapper;
 public:
  static const unsigned int full = 1;
  static const unsigned int eltSize = 1;
  static const unsigned int slotSize = sizeof(unsigned int);
  static constexpr unsigned int slotElts = 8 * slotSize;

  BV(unsigned int len, bool slotWise = false);
  BV(const std::vector<unsigned int> &_raw);
  BV(unsigned int _raw[], size_t _nSlot);
  BV(std::vector<unsigned int> &_raw, unsigned int _nSlot);

  ~BV();

  /**
     @brief Accessor for position within the 'raw' buffer.
   */
  inline unsigned int *Raw(unsigned int off) {
    return raw + off;
  }

  void Consume(std::vector<unsigned int> &out, unsigned int bitEnd = 0) const;
  unsigned int PopCount() const;

  BV *Resize(unsigned int bitMin);

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

     @return length of containing aligned quantity in units of buffer type.
   */
  static inline unsigned int SlotAlign(unsigned int len) {
    return (len + slotElts - 1) / slotElts;
  }


  static inline unsigned int Stride(unsigned int len) {
    return slotElts * SlotAlign(len);
  }

  
  /**
   */
  static unsigned int SlotMask(unsigned int pos, unsigned int &mask) {
    unsigned int slot = pos / slotElts;
    mask = full << (pos - (slot * slotElts));

    return slot;
  }


  bool Test(unsigned int slot, unsigned int mask) const {
    return (raw[slot] & mask) != 0;
  }

  
  /**
     @brief Tests the bit at a specified position.

     @param bv is the bit vector implementation.

     @param pos is the bit position to test.

     @return true iff bit position is set in the bit vector.
   */
  inline bool TestBit(unsigned int pos) const {
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
  inline void SetBit(unsigned int pos, bool on = true) {
    unsigned int slot = pos / slotElts;
    unsigned int mask = full << (pos - (slot * slotElts));
    unsigned int val = raw[slot];
    raw[slot] = on ? val | mask : val & ~mask;
  }


  inline unsigned int Slot(unsigned int slot) const {
    return raw[slot];
  }
  
  
  inline void SetSlot(unsigned int slot, unsigned int val) {
    raw[slot] = val;
  }


  inline void Clear() {
    for (unsigned int i = 0; i < nSlot; i++) {
      raw[i] = 0;
    }
  }
};


class BitRow : public BV {
 public:
 BitRow(unsigned int *_raw, unsigned int _nSlot) : BV(_raw, _nSlot) {}
  ~BitRow() {
  }
};


/**
   @brief Like a bit vector, but with row-major strided access.

 */
class BitMatrix : public BV {
  const unsigned int nRow;
  const unsigned int stride;
  void Export(unsigned int _nRow, std::vector<std::vector<unsigned int> > &bmOut);
  void ColExport(unsigned int _nRow, std::vector<unsigned int> &outCol, unsigned int colIdx);

 public:
  BitMatrix(unsigned int _nRow, unsigned int _nCol);
  BitMatrix(unsigned int _nRow, unsigned int _nCol, const std::vector<unsigned int> &_raw);
  BitMatrix(std::vector<unsigned int> &_raw, unsigned int _nRow, unsigned int _nCol);
  BitMatrix(unsigned int _raw[], size_t _nRow, size_t _nCol);
  ~BitMatrix();

  inline unsigned int NRow() const {
    return nRow;
  }
  
  static void Export(const std::vector<unsigned int> &_raw, unsigned int _nRow, std::vector<std::vector<unsigned int> > &vecOut);


  inline BitRow *Row(unsigned int row) {
    return new BitRow(Raw((stride * row)/slotElts), stride);
  }


  /**
     @brief Bit test with short-circuit for zero-length matrix.

     @return whether bit at specified coordinate is set.
   */
  inline bool TestBit(unsigned int row, unsigned int col) const {
    return stride == 0 ? false : BV::TestBit(row * stride + col);
  }

  
  inline void SetBit(unsigned int row, unsigned int col, bool on = true) {
    BV::SetBit(row * stride + col, on);
  }


  inline void ClearBit(unsigned int row, unsigned int col) {
    SetBit(row, col, false);
  }
};


/**
   @brief Jagged bit matrix:  unstrided access.
 */
class BVJagged : public BV {
  const size_t nElt;
  const unsigned int *rowOrigin;
  const unsigned int nRow;
  void Export(std::vector<std::vector<unsigned int> > &outVec);
  void RowExport(std::vector<unsigned int> &outRow, unsigned int rowHeight, unsigned int rowIdx) const;
  unsigned int RowHeight(unsigned int rowIdx) const;
 public:
  BVJagged(unsigned int _raw[], size_t _nSlot, const unsigned int _origin[], unsigned int _nRow);
  ~BVJagged();
  static void Export(unsigned int _raw[], std::size_t facLen, const unsigned int _origin[], unsigned int _nElt, std::vector<std::vector<unsigned int> > &outVec);


  inline size_t NElt() const {
    return nElt;
  }


  /**
     @brief Bit test for jagged matrix.

     @param row is the (nonstrided) row.

     @param pos is the bit position within the row.

     @return true iff bit set.

   */
  inline bool TestBit(unsigned int row, unsigned int pos) const {
    unsigned int mask;
    unsigned int slot = SlotMask(pos, mask);
    unsigned int base = rowOrigin[row];
    
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
  CharV(unsigned int *_raw, unsigned int _nSlot);
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

     @return length of containing quantity in units of buffer type..
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
     @brief Sets slot at position to value passed.

     @param pos is the position to set.

     @return void, with side-effected raw value.
   */
  inline void Set(unsigned int pos, unsigned char val) {
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
  CharRow(unsigned int *_raw, unsigned int _nSlot) : CharV(_raw, _nSlot) {}
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
