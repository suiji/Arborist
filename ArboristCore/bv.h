// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bv.h

   @brief Methods for manipulating hand-rolled bit vector.

   @author Mark Seligman

 */

#ifndef ARBORIST_BV_H
#define ARBORIST_BV_H

#include <vector>

class BV {
  unsigned int *raw;
  const unsigned int nSlot;
  const unsigned int nBit;
  static const unsigned int slotBits = 8 * sizeof(unsigned int);
 public:
  BV(unsigned int len, bool slotWise = false);
  BV(const std::vector<unsigned int> &_raw);
  ~BV();

  inline unsigned int NBit() const {
    return nBit;
  }
  

  void Consume(std::vector<unsigned int> &out, unsigned int bitEnd = 0) const;

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
  static unsigned int SlotBits() {
    return slotBits;
  }
  
  // Compiler should be generating shifts.  In c++11 can replace
  // with constant expression for log2(slotBits) and introduce
  // shifts explicitly.

  /**
     @brief Aligns length to the nearest containing bit slot.

     @param len is the length to align.

     @return length of containing quantity.
   */
  static inline unsigned int SlotAlign(unsigned int len) {
    return (len + slotBits - 1) / slotBits;
  }

  
  /**
   */
  static unsigned int SlotMask(unsigned int pos, unsigned int &mask) {
    unsigned int slot = pos / slotBits;
    mask = 1 << (pos - (slot * slotBits));

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
  inline bool IsSet(unsigned int pos) const {
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
  inline void SetBit(unsigned int pos) {
    unsigned int slot = pos / slotBits;
    unsigned int mask = 1 << (pos - (slot * slotBits));
    raw[slot] |= mask;
  }


  inline unsigned int Slot(unsigned int slot) const {
    return raw[slot];
  }
  
  
  inline void SetSlot(unsigned int slot, unsigned int val) {
    raw[slot] = val;
  }

};


/**
   @brief Like a bit vector with strided access.

 */
class BitMatrix : public BV {
  const unsigned int nRow;
  const unsigned int stride;
  void Export(unsigned int _nRow, std::vector<std::vector<unsigned int> > &bmOut);
  void ColExport(unsigned int _nRow, std::vector<unsigned int> &outCol, unsigned int colIdx);

 public:
  BitMatrix(unsigned int _nRow, unsigned int _nCol);
  BitMatrix(unsigned int _nRow, unsigned int _nCol, const std::vector<unsigned int> &_raw);
  ~BitMatrix();

  void SetColumn(const class BV *vec, int colIdx);

  static void Export(const std::vector<unsigned int> &_raw, unsigned int _nRow, std::vector<std::vector<unsigned int> > &vecOut);

  /**
     @brief Bit test with short-circuit for zero-length matrix.

     @return whether bit at specified coordinate is set.
   */
  inline bool IsSet(unsigned int row, unsigned int col) const {
    return stride == 0 ? false : BV::IsSet(row * stride + col);
  }

  inline void SetBit(unsigned int row, unsigned int col) {
    BV::SetBit(row * stride + col);
  }

};


/**
   @brief Jagged bit matrix:  unstrided access.
 */
class BVJagged : public BV {
  const unsigned int nRow;
  unsigned int *rowOrigin;
  void Export(std::vector<std::vector<unsigned int> > &outVec);
  void RowExport(std::vector<unsigned int> &outRow, unsigned int rowHeight, unsigned int rowIdx) const;
  unsigned int RowHeight(unsigned int rowIdx) const;
 public:
  BVJagged(const std::vector<unsigned int> &_raw, const std::vector<unsigned int> _origin);
  ~BVJagged();
  static void Export(const std::vector<unsigned int> _origin, const std::vector<unsigned int> _raw, std::vector<std::vector<unsigned int> > &outVec);


  /**
     @brief Bit test for jagged matrix.

     @param row is the (nonstrided) row.

     @param pos is the bit position within the row.

     @return true iff bit set.

   */
  inline bool IsSet(unsigned int row, unsigned int pos) const {
    unsigned int mask;
    unsigned int slot = SlotMask(pos, mask);
    unsigned int base = rowOrigin[row];
    
    return Test(base + slot, mask);
  }
};


#endif
