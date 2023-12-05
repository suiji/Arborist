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
typedef size_t BVSlotT; // Slot container type.


class BV {
  size_t nSlot; ///< Number of typed (BVSlotT) slots.
  vector<BVSlotT> rawV; ///< Internal manager for writable instances.
  
 public:
  static const size_t full;
  static const size_t allOnes;
  static const size_t slotSize;
  static const size_t slotElts;

  BV(const BV* bv) :
    nSlot(bv->nSlot),
    rawV(vector<BVSlotT>(nSlot)) {
  }

  
  BV(size_t bitLen) :
    nSlot(slotAlign(bitLen)),
    rawV(vector<BVSlotT>(nSlot)) {
  }

  /**
     @brief Slotwise initialization from constant vector.
  */
  BV(const vector<BVSlotT>& raw_) :
    nSlot(raw_.size()),
    rawV(raw_.begin(), raw_.end()) {
  }

  
  /**
     @brief Bytewise initialization from constant buffer.
   */
  BV(const unsigned char bytes[],
     size_t nSlot_) :
    nSlot(nSlot_),
    rawV(vector<BVSlotT>(nSlot)) {
    if (nSlot != 0) {
      unsigned char* bufOut = reinterpret_cast<unsigned char*>(&rawV[0]);
      for (size_t idx = 0; idx < nSlot * sizeof(BVSlotT); idx++) {
	*bufOut++ = *bytes++;
      }
    }
  }


  ~BV() = default;


  const BVSlotT& getRaw(size_t i) const {
    return rawV[i];
  }
  
  
  /**
     @brief Sets slots from a vector position deltas.
   */
  void delEncode(const vector<IndexT>& delPos);


  void dumpRaw(unsigned char *bbRaw) const {
    if (nSlot == 0)
      return;
    const unsigned char* rawChar = reinterpret_cast<const unsigned char*>(&rawV[0]);
    for (size_t i = 0; i < nSlot * sizeof(BVSlotT); i++) {
      bbRaw[i] = rawChar[i];
    }
  }

  
  vector<BVSlotT> dumpVec(size_t base, size_t extent) const {
    vector<BVSlotT> outVec(extent);
    IndexT idx = 0;
    for (auto & cell : outVec) {
      cell = getRaw(base + idx++);
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
     @brief Appends whole slots onto output vector, preserving endianness.

     @param[out] out outputs the raw bit vector contents.

     @param bitEnd specifies the ending bit position.

     @retun number of native slots consumed.
  */
  size_t appendSlots(vector<BVSlotT>& out,
		     size_t bitEnd) const {
    size_t slotEnd = slotAlign(bitEnd);
    out.insert(out.end(), rawV.begin(), rawV.begin() + slotEnd);
    return slotEnd;
  }


  BV operator|(const BV& bvR) {
    /*
    if (bvR.getNSlot() != nSlot) {
      throw std::invalid_argument("mismatched bit vector | operation");
      }*/
    BV bvOr(this);
    for (size_t i = 0; i < nSlot; i++) {
      bvOr.rawV[i] = getRaw(i) | bvR.getRaw(i);
    }
    return bvOr;
  }

  
  BV& operator&=(const BV& bvR) {
    /*
    if (nSlot != bvR.getNSlot()) {
      throw std::invalid_argument("mismatched bit vector &= operation");
      }*/

    for (size_t i = 0; i < nSlot; i++) {
      rawV[i] &= bvR.getRaw(i);
    }
    return *this;
  }


  BV& operator|=(const BV& bvR) {
    /*
    if (nSlot != bvR.getNSlot()) {
      throw std::invalid_argument("mismatched bit vector |= operation");
      }*/

    for (size_t i = 0; i < nSlot; i++) {
      rawV[i] |= bvR.getRaw(i);
    }
    return *this;
  }


  BV operator~() {
    BV bvTilde(this);
    for (size_t i = 0; i < nSlot; i++) {
      bvTilde.rawV[i] = ~getRaw(i);
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
  static size_t slotAlign(size_t len) {
    return (len + slotElts - 1) / slotElts;
  }


  static size_t strideBytes(size_t len) {
    return slotAlign(len) * sizeof(BVSlotT);
  }

  /**
     @return length of aligned row in bits.
   */
  static size_t Stride(size_t len) {
    return slotElts * slotAlign(len);
  }

  
  /**
     @brief Builds a bit mask having a single bit high.

     @param pos is a bit position

     @param mask outputs a slot-width mask with the bit at 'pos' set.

     @return slot containing position.
   */
  static size_t slotMask(size_t pos,
				BVSlotT& mask) {
    size_t slot = pos / slotElts;
    mask = full << (pos - (slot * slotElts));

    return slot;
  }


  bool test(size_t slot, BVSlotT mask) const {
    return (getRaw(slot) & mask) == mask;
  }

  
  /**
     @brief Tests the bit at a specified position.

     @param bv is the bit vector implementation.

     @param pos is the bit position to test.

     @return true iff bit position is set in the bit vector.
   */
  bool testBit(size_t pos) const {
    BVSlotT mask;
    size_t slot = slotMask(pos, mask);

    return test(slot, mask);
  }

  
  /**
     @brief Sets the bit at position 'pos'.

     @param pos is the position to set.

     @param on indicates whether to set the bit on/off.
   */
  void setBit(size_t pos,
		     bool on = true) {
    BVSlotT mask;
    size_t slot = slotMask(pos, mask);
    BVSlotT val = rawV[slot];
    rawV[slot] = on ? (val | mask) : (val & ~mask);
  }

  
  void setSlot(size_t slot, BVSlotT val) {
    rawV[slot] = val;
  }

  /**
     @brief Sets all slots to zero.
   */
  void clear() {
    fill(rawV.begin(), rawV.end(), 0ul);
  }


  /**
     @brief Sets all slots high.
   */
  void saturate() {
    fill(rawV.begin(), rawV.end(), allOnes);
  }
};


/**
   @brief Like a bit vector, but with row-major strided access.

 */
class BitMatrix : public BV {
  const unsigned int nRow;
  const IndexT stride; // Number of uint cells per row.

  /**
     @brief Exports matrix as vector of column vectors.

     @param _nRow is the external row count.

     @return void, with output reference parameter.
  */
  void dump(unsigned int nRow_,
	    vector<vector<BVSlotT>>& bmOut) const;


  /**
     @brief Exports an individual column to a uint vector.

   @param nRow is the external row count.

   @param outCol outputs the column.

   @param colIdx is the column index.

   @return void, with output reference vector.
  */
  void colDump(unsigned int nRow_,
               vector<BVSlotT>& outCol,
               IndexT colIdx) const;

 public:
  BitMatrix(unsigned int nRow_,
	    IndexT nCol_);

  
  BitMatrix(const BVSlotT raw_[],
	    unsigned int nRow_,
	    IndexT nCol_);

  
  ~BitMatrix();


  auto getNRow() const {
    return nRow;
  }


  size_t getStride() const {
    return stride;
  }

  
  /**
     @brief Bit test with short-circuit for zero-length matrix.

     @return whether bit at specified coordinate is set.
   */
  bool testBit(unsigned int row, IndexT col) const {
    return stride == 0 ? false : BV::testBit(row * stride + col);
  }

  
  void setBit(unsigned int row,
		     IndexT col,
		     bool on = true) {
    BV::setBit(row * stride + col, on);
  }


  void clearBit(unsigned int row, IndexT col) {
    setBit(row, col, false);
  }
};


/**
   @brief Jagged bit matrix.
 */
class BVJagged : public BV {
  const vector<size_t> rowHeight;
  const unsigned int nRow;

public:
  BVJagged(const BVSlotT raw_[],
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
  bool testBit(size_t row, size_t pos) const {
    BVSlotT mask;
    size_t slot = slotMask(pos, mask);
    size_t base = row == 0 ? 0 : rowHeight[row-1];
    
    return test(base + slot, mask);
  }


  /**
     @brief Dumps each row into a separate vector.
   */
  vector<vector<BVSlotT>> dump() const;


  /**
     @brief Outputs a row of bits as a packed integer vector.
   */
  vector<BVSlotT> rowDumpRaw(size_t rowIdx) const;
  
};

#endif
