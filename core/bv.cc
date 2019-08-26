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
BV::BV(size_t len, bool slotWise) : nSlot(slotWise ? len : slotAlign(len)), raw(new RawT[nSlot]), wrapper(false) {
  clear();
}


/**
   @brief Copies contents of constant vector.
 */
BV::BV(const vector<RawT> &raw_) : nSlot(raw_.size()), raw(new RawT[nSlot]), wrapper(false) {
  for (size_t i = 0; i < nSlot; i++) {
    raw[i] = raw_[i];
  }
}


/**
   @brief Wrapper constructor.  Initializes external container if empty.
 */
BV::BV(vector<RawT>& raw_, size_t _nSlot) : nSlot(_nSlot), wrapper(true) {
  if (raw_.size() == 0) {
    for (size_t slot = 0; slot < nSlot; slot++) {
      raw_.push_back(0);
    }
  }
  raw = &raw_[0];
}


BV::BV(RawT raw_[], size_t nSlot_) : nSlot(nSlot_), raw(raw_), wrapper(true) {
}


/**
 */
BV::~BV() {
  if (!wrapper)
    delete [] raw;
}


/**
 @brief Resizes to next power of two, if needed.

 @param bitMin is the minimum count of raw bits.

 @return resized vector or this.
*/
BV *BV::Resize(size_t bitMin) {
  size_t slotMin = slotAlign(bitMin);
  if (nSlot >= slotMin)
    return this;

  size_t slotsNext = nSlot;
  while (slotsNext < slotMin)
    slotsNext <<= 1;
  
  BV *bvNew = new BV(slotsNext, true);
  for (size_t i = 0; i < nSlot; i++) {
    bvNew->raw[i] = raw[i];
  }

  delete this;

  return bvNew;
}


void BV::consume(vector<RawT> &out, size_t bitEnd) const {
  size_t slots = bitEnd == 0 ? nSlot : slotAlign(bitEnd);
  out.reserve(slots);
  out.insert(out.end(), raw, raw + slots);
}


/**
 */
BitMatrix::BitMatrix(size_t _nRow, unsigned int _nCol) : BV(_nRow * Stride(_nCol)), nRow(_nRow), stride(Stride(_nCol)) {
}


/**
   @brief Copy constructor.  Sets stride to zero if empty.
 */
BitMatrix::BitMatrix(size_t _nRow, unsigned int _nCol, const vector<unsigned int> &raw_) : BV(raw_), nRow(_nRow), stride(raw_.size() > 0 ? Stride(_nCol) : 0) {
}


BitMatrix::BitMatrix(RawT raw_[], size_t _nRow, size_t _nCol) : BV(raw_, _nRow * Stride(_nCol)), nRow(_nRow), stride(nRow > 0 ? Stride(_nCol) : 0) {
}


/**
 */
BitMatrix::~BitMatrix() {
}


/**
 */
BVJagged::BVJagged(RawT raw_[],
                   const unsigned int rowExtent_[],
                   size_t nRow_) : BV(raw_, rowExtent_[nRow_-1]),
                                         rowExtent(rowExtent_),
                                         nRow(nRow_) {
}


/**
 */
BVJagged::~BVJagged() {
}


/**
   @brief Exports contents of a forest.
 */
void BVJagged::dump(vector<vector<unsigned int> > &outVec) {
  for (IndexT row = 0; row < nRow; row++) {
    outVec[row] = rowDump(row);
  }
}


/**
   @brief Exports contents for an individual row.
 */
vector<unsigned int> BVJagged::rowDump(size_t rowIdx) const {
  vector<unsigned int> outVec(rowExtent[rowIdx]);
  for (IndexT idx = 0; idx < outVec.size(); idx++) {
    outVec[idx] = testBit(rowIdx, idx);
  }
  return outVec;
}


/**
   @brief Static entry.  Exports matrix as vector of vectors.

   @param outBag outputs the columns as vectors.

   @return void, with output vector parameter.
 */
void BitMatrix::dump(const vector<unsigned int> &raw_, size_t _nRow, vector<vector<unsigned int> > &vecOut) {
  unsigned int _nCol = vecOut.size();
  BitMatrix *bm = new BitMatrix(_nRow, _nCol, raw_);
  bm->dump(_nRow, vecOut);

  delete bm;
}


/**
   @brief Exports matrix as vector of column vectors.

   @param _nRow is the external row count.

   @return void, with output reference parameter.
 */
void BitMatrix::dump(size_t nRow_, vector<vector<unsigned int> > &outCol) const {
  for (size_t i = 0; i < stride; i++) {
    outCol[i] = vector<unsigned int>(nRow_);
    colDump(nRow_, outCol[i], i);
  }
}


/**
   @brief Exports an individual column to a uint vector.

   @param _nRow is the external row count.

   @param outCol outputs the column.

   @param colIdx is the column index.

   @return void, with output reference vector.
 */
void BitMatrix::colDump(size_t _nRow, vector<unsigned int> &outCol, unsigned int colIdx) const {
  for (size_t row = 0; row < _nRow; row++)
    outCol[row] = testBit(row, colIdx) ? 1 : 0;
}
