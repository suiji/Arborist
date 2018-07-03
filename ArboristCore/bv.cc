// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bv.cc

   @brief Methods for manipulating hand-rolled bit vector.

   @author Mark Seligman
 */

#include "bv.h"


/**
 */
BV::BV(size_t len, bool slotWise) : nSlot(slotWise ? len : SlotAlign(len)), raw(new unsigned int[nSlot]), wrapper(false) {
  for (unsigned int i = 0; i < nSlot; i++) {
    raw[i] = 0;
  }
}


/**
   @brief Copies contents of constant vector.
 */
BV::BV(const vector<unsigned int> &raw_) : nSlot(raw_.size()), raw(new unsigned int[nSlot]), wrapper(false) {
  for (unsigned int i = 0; i < nSlot; i++) {
    raw[i] = raw_[i];
  }
}


/**
   @brief Wrapper constructor.  Initializes external container if empty.
 */
BV::BV(vector<unsigned int> &raw_, unsigned int _nSlot) : nSlot(_nSlot), wrapper(true) {
  if (raw_.size() == 0) {
    for (unsigned int slot = 0; slot < nSlot; slot++) {
      raw_.push_back(0);
    }
  }
  raw = &raw_[0];
}


BV::BV(unsigned int raw_[], size_t nSlot_) : nSlot(nSlot_), raw(raw_), wrapper(true) {
}


/**
 */
BV::~BV() {
  if (!wrapper)
    delete [] raw;
}


CharV::CharV(unsigned int len) : nSlot(SlotAlign(len)), wrapper(false) {
  raw = new unsigned int[nSlot];
  for (unsigned int i = 0; i < nSlot; i++)
    raw[i] = 0;
}


CharV::CharV(unsigned int *raw_, unsigned int _nSlot) : raw(raw_), nSlot(_nSlot), wrapper(true) {
}


CharV::~CharV() {
  if (!wrapper)
    delete [] raw;
}


/**
 @brief Resizes to next power of two, if needed.

 @param bitMin is the minimum count of raw bits.

 @return resized vector or this.
*/
BV *BV::Resize(size_t bitMin) {
  unsigned int slotMin = SlotAlign(bitMin);
  if (nSlot >= slotMin)
    return this;

  unsigned int slotsNext = nSlot;
  while (slotsNext < slotMin)
    slotsNext <<= 1;
  
  BV *bvNew = new BV(slotsNext, true);
  for (unsigned int i = 0; i < nSlot; i++) {
    bvNew->raw[i] = raw[i];
  }

  delete this;

  return bvNew;
}


/**
   @brief Appends contents onto output vector.

   @return void, with output vector parameter.
 */
void BV::Consume(vector<unsigned int> &out, unsigned int bitEnd) const {
  unsigned int slots = bitEnd == 0 ? nSlot : SlotAlign(bitEnd);
  out.reserve(slots);
  out.insert(out.end(), raw, raw + slots);
}


unsigned int BV::PopCount() const {
  unsigned int pop = 0;
  for (unsigned int i = 0; i < nSlot; i++) {
    unsigned int val = raw[i];
    for (unsigned int j = 0; j < 8 * sizeof(unsigned int); j++) {
      pop += val & 1;
      val >>= 1;
    }
  }
  return pop;
}


/**
 */
BitMatrix::BitMatrix(unsigned int _nRow, unsigned int _nCol) : BV(_nRow * Stride(_nCol)), nRow(_nRow), stride(Stride(_nCol)) {
}


/**
   @brief Copy constructor.  Sets stride to zero if empty.
 */
BitMatrix::BitMatrix(unsigned int _nRow, unsigned int _nCol, const vector<unsigned int> &raw_) : BV(raw_), nRow(_nRow), stride(raw_.size() > 0 ? Stride(_nCol) : 0) {
}


BitMatrix::BitMatrix(unsigned int raw_[], size_t _nRow, size_t _nCol) : BV(raw_, _nRow * Stride(_nCol)), nRow(_nRow), stride(nRow > 0 ? Stride(_nCol) : 0) {
}


/**
 */
BitMatrix::~BitMatrix() {
}


/**
 */
BVJagged::BVJagged(unsigned int raw_[], size_t _nSlot, const unsigned int _rowOrigin[], unsigned int _nRow) : BV(raw_, _nSlot), nElt(_nSlot * slotElts), rowOrigin(_rowOrigin), nRow(_nRow) {
}


/**
 */
BVJagged::~BVJagged() {
}


/**
 */
unsigned int BVJagged::RowHeight(unsigned int rowIdx) const {
  if (rowIdx < nRow - 1) {
    return sizeof(unsigned int) * (rowOrigin[rowIdx + 1] - rowOrigin[rowIdx]);
  }
  else {
    return NElt() - sizeof(unsigned int) * rowOrigin[rowIdx];
  }
}

/**
   @brief Exports contents of a forest.
 */
void BVJagged::Export(vector<vector<unsigned int> > &outVec) {
  for (unsigned int row = 0; row < nRow; row++) {
    unsigned int rowHeight = RowHeight(row);
    outVec[row] = vector<unsigned int>(rowHeight);
    RowExport(outVec[row], rowHeight, row);
  }
}


/**
   @brief Exports contents for an individual row.
 */
void BVJagged::RowExport(vector<unsigned int> &outRow, unsigned int rowHeight, unsigned int rowIdx) const {
  for (unsigned int idx = 0; idx < rowHeight; idx++) {
    outRow[idx] = testBit(rowIdx, idx);
  }
}


/**
   @brief Static entry.  Exports matrix as vector of vectors.

   @param outBag outputs the columns as vectors.

   @return void, with output vector parameter.
 */
void BitMatrix::Export(const vector<unsigned int> &raw_, unsigned int _nRow, vector<vector<unsigned int> > &vecOut) {
  unsigned int _nCol = vecOut.size();
  BitMatrix *bm = new BitMatrix(_nRow, _nCol, raw_);
  bm->Export(_nRow, vecOut);

  delete bm;
}


/**
   @brief Exports matrix as vector of column vectors.

   @param _nRow is the external row count.

   @return void, with output reference parameter.
 */
void BitMatrix::Export(unsigned int nRow_, vector<vector<unsigned int> > &outCol) const {
  for (unsigned int i = 0; i < stride; i++) {
    outCol[i] = vector<unsigned int>(nRow_);
    ColExport(nRow_, outCol[i], i);
  }
}


/**
   @brief Exports an individual column to a uint vector.

   @param _nRow is the external row count.

   @param outCol outputs the column.

   @param colIdx is the column index.

   @return void, with output reference vector.
 */
void BitMatrix::ColExport(unsigned int _nRow, vector<unsigned int> &outCol, unsigned int colIdx) const {
  for (unsigned int row = 0; row < _nRow; row++)
    outCol[row] = testBit(row, colIdx) ? 1 : 0;
}


CharMatrix::CharMatrix(unsigned int nRow, unsigned int _nCol) : CharV(SlotAlign(nRow) * _nCol), stride(Stride(nRow)), nCol(_nCol) {
}
