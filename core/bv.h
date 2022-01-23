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
  size_t nSlot; // Number of typed (RawT) slots.
  vector<RawT> rawV; // Internal manager for writable instances.
  const RawT* raw; // Points to rawV iff writable, else external buffer.
  
 public:
  static constexpr unsigned int allOnes = 0xffffffff;
  static constexpr unsigned int full = 1;
  static constexpr unsigned int eltSize = 1;
  static constexpr size_t slotSize = sizeof(RawT);
  static constexpr size_t slotElts = 8 * slotSize; // # bits in slot.

  BV(const BV* bv) :
    nSlot(bv->nSlot),
    rawV(vector<RawT>(nSlot)),
    raw(nSlot == 0 ? nullptr : &rawV[0]) {
  }

  
  BV(size_t bitLen) :
    nSlot(slotAlign(bitLen)),
    rawV(vector<RawT>(nSlot)),
    raw(nSlot == 0 ? nullptr : &rawV[0]) {
  }

  /**
     @brief Copies contents of constant vector.
  */
  BV(const vector<RawT>& raw_) :
    nSlot(raw_.size()),
    rawV(raw_.begin(), raw_.end()),
    raw(nSlot == 0 ? nullptr : &rawV[0]) {
  }


  /**
     @brief Wraps external buffer:  unwritable.

     @param raw_ points to an external buffer.

     @param nSlot_ is the number of readable RawT slots in the buffer.
   */
   BV(const RawT raw_[],
      size_t nSlot_) : nSlot(nSlot_),
		       raw(raw_) {  }

  
  ~BV(){}

  
  /**
     @brief Sets slots from a vector position deltas.
   */
  void delEncode(const vector<IndexT>& delPos);
  
  inline void dumpRaw(unsigned char *bbRaw) const {
    for (size_t i = 0; i < nSlot * sizeof(RawT); i++) {
      bbRaw[i] = reinterpret_cast<const unsigned char*>(&raw[0])[i];
    }
  }

  
  inline vector<RawT> dumpVec(size_t base, size_t extent) const {
    vector<RawT> outVec(extent);
    IndexT idx = 0;
    for (auto & cell : outVec) {
      cell = raw[base + idx++];
    }
    return outVec;
  }


  /**
     @brief Determines whether container is empty.
   */
  bool isEmpty() const {
    return nSlot == 0;
  }

  
  /**
     @brief Appends whole slots onto output vector.

     @param[out] out outputs the raw bit vector contents.

     @param bitEnd specifies a known end position.

     @retun number of native slots consumed.
  */
  size_t appendSlots(vector<RawT>& out, size_t bitEnd) const {
    size_t slotEnd = slotAlign(bitEnd);
    out.insert(out.end(), raw, raw + slotEnd);
    return slotEnd;
  }


  BV operator|(const BV& bvR) {
    /*
    if (bvR.getNSlot() != nSlot) {
      throw std::invalid_argument("mismatched bit vector | operation");
      }*/
    BV bvOr(this);
    for (size_t i = 0; i < nSlot; i++) {
      bvOr.rawV[i] = raw[i] | bvR.raw[i];
    }
    return bvOr;
  }

  
  BV& operator&=(const BV& bvR) {
    /*
    if (nSlot != bvR.getNSlot()) {
      throw std::invalid_argument("mismatched bit vector &= operation");
      }*/

    for (size_t i = 0; i < nSlot; i++) {
      rawV[i] &= bvR.raw[i];
    }
    return *this;
  }


  BV& operator|=(const BV& bvR) {
    /*
    if (nSlot != bvR.getNSlot()) {
      throw std::invalid_argument("mismatched bit vector |= operation");
      }*/

    for (size_t i = 0; i < nSlot; i++) {
      rawV[i] |= bvR.raw[i];
    }
    return *this;
  }


  BV operator~() {
    BV bvTilde(this);
    for (size_t i = 0; i < nSlot; i++) {
      bvTilde.rawV[i] = ~raw[i];
    }
    return bvTilde;
  }

  
  /**
     @brief Resizes to accommodate desired bit size.

     N.B.:  Should not be used for wrappers, as external vector
     not copied.

     @param bitMin is the minimum count of raw bits.
  */
  void resize(size_t bitMin);

  
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

     @param pos is the position to set.

     @param on indicates whether to set the bit on/off.
   */
  inline void setBit(size_t pos,
		     bool on = true) {
    RawT mask;
    size_t slot = slotMask(pos, mask);
    RawT val = rawV[slot];
    rawV[slot] = on ? (val | mask) : (val & ~mask);
  }

  
  inline void setSlot(size_t slot, RawT val) {
    rawV[slot] = val;
  }

  /**
     @brief Sets all slots to zero.
   */
  inline void clear() {
    fill(rawV.begin(), rawV.end(), 0ul);
  }


  /**
     @brief Sets all slots high.
   */
  inline void saturate() {
    fill(rawV.begin(), rawV.end(), allOnes);
  }
};


/**
   @brief Like a bit vector, but with row-major strided access.

 */
class BitMatrix : public BV {
  const size_t nRow;
  const unsigned int stride; // Number of uint cells per row.

  /**
     @brief Exports matrix as vector of column vectors.

     @param _nRow is the external row count.

     @return void, with output reference parameter.
  */
  void dump(size_t _nRow, vector<vector<unsigned int> > &bmOut) const;


  /**
     @brief Exports an individual column to a uint vector.
   @param _nRow is the external row count.

   @param outCol outputs the column.

   @param colIdx is the column index.

   @return void, with output reference vector.
  */
  void colDump(size_t _nRow,
               vector<unsigned int> &outCol,
               unsigned int colIdx) const;

 public:
  BitMatrix(size_t _nRow, unsigned int _nCol);

  BitMatrix(const RawT raw_[], size_t _nRow, size_t _nCol);

  ~BitMatrix();

  inline size_t getNRow() const {
    return nRow;
  }


  inline size_t getStride() const {
    return stride;
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
   @brief Jagged bit matrix, caches extent vector.
 */
class BVJagged : public BV {
  const vector<size_t> rowHeight;
  const size_t nRow;

public:
  BVJagged(const RawT raw_[],
           const vector<size_t>& height); // Cumulative extent per row.

  ~BVJagged();

  
  size_t getRowHeight(size_t row) const {
    return rowHeight[row];
  }
  
  
  /**
     @brief Bit test for jagged matrix.

     @param row is the (nonstrided) row.

     @param pos is the bit position within the row.

     @return true iff bit set.

   */
  inline bool testBit(size_t row, size_t pos) const {
    RawT mask;
    size_t slot = slotMask(pos, mask);
    unsigned int base = row == 0 ? 0 : rowHeight[row-1];
    
    return test(base + slot, mask);
  }


  /**
     @brief Dumps each row into a separate vector.
   */
  vector<vector<RawT>> dump() const;


  /**
     @brief Outputs a row of bits as a packed integer vector.
   */
  vector<RawT> rowDumpRaw(size_t rowIdx) const;
  
};

#endif
