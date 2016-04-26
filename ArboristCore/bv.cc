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

//#include <iostream>
using namespace std;


/**
 */
BV::BV(unsigned int len, bool slotWise) : nSlot(slotWise ? len : SlotAlign(len)), nBit(slotWise ? len * slotBits : len) {
  raw = new unsigned int[nSlot];
  for (unsigned int i = 0; i < nSlot; i++) {
    raw[i] = 0;
  }
}


/**
 */
BV::BV(const std::vector<unsigned int> &_raw) : nSlot(_raw.size()), nBit(nSlot * slotBits) {
  raw = new unsigned int[nSlot];
  for (unsigned int i = 0; i < nSlot; i++) {
    raw[i] = _raw[i];
  }
}


/**
 */
BV::~BV() {
  delete [] raw;
}


/**
 @brief Resizes to next power of two, if needed.

 @param bitMin is the minimum count of raw bits.

 @return resized vector or this.
*/
BV *BV::Resize(unsigned int bitMin) {
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
void BV::Consume(std::vector<unsigned int> &out, unsigned int bitEnd) const {
  unsigned int slots = bitEnd == 0 ? nSlot : SlotAlign(bitEnd);
  out.reserve(slots);
  out.insert(out.end(), raw, raw + slots);
}


/**
 */
BitMatrix::BitMatrix(unsigned int _nRow, unsigned int _nCol) : BV(_nRow * _nCol), nRow(_nRow), stride(_nCol) {
}


/**
   @brief Constructor.  Sets stride to zero if empty.
 */
BitMatrix::BitMatrix(unsigned int _nRow, unsigned int _nCol, const std::vector<unsigned int> &_raw) : BV(_raw), nRow(_nRow), stride(_raw.size() > 0 ? _nCol : 0) {
}


/**
 */
BitMatrix::~BitMatrix() {
}


/**
 */
BVJagged::BVJagged(const std::vector<unsigned int> &_raw, const std::vector<unsigned int> _rowOrigin) : BV(_raw), nRow(_rowOrigin.size()) {
  rowOrigin = new unsigned int[nRow];
  for (unsigned int i = 0; i < nRow; i++) {
    rowOrigin[i] = _rowOrigin[i];
  }
}


/**
 */
BVJagged::~BVJagged() {
  delete [] rowOrigin;
}


/**
 */
unsigned int BVJagged::RowHeight(unsigned int rowIdx) const {
  if (rowIdx < nRow - 1) {
    return sizeof(unsigned int) * (rowOrigin[rowIdx + 1] - rowOrigin[rowIdx]);
  }
  else {
    return NBit() - sizeof(unsigned int) * rowOrigin[rowIdx];
  }
}


void BVJagged::Export(const std::vector<unsigned int> _origin, const std::vector<unsigned int> _raw, std::vector<std::vector<unsigned int> > &outVec) {
  BVJagged *bvj = new BVJagged(_raw, _origin);
  bvj->Export(outVec);

  delete bvj;
}

/**
   @brief Exports contents of a forest.
 */
void BVJagged::Export(std::vector<std::vector<unsigned int> > &outVec) {
  for (unsigned int row = 0; row < nRow; row++) {
    unsigned int rowHeight = RowHeight(row);
    outVec[row] = std::vector<unsigned int>(rowHeight);
    RowExport(outVec[row], rowHeight, row);
  }
}


/**
   @brief Exports contents for an individual row.
 */
void BVJagged::RowExport(std::vector<unsigned int> &outRow, unsigned int rowHeight, unsigned int rowIdx) const {
  for (unsigned int idx = 0; idx < rowHeight; idx++) {
    outRow[idx] = IsSet(rowIdx, idx);
  }
}


/**
  @brief Transfers vector of bits to matrix column.

  @param vec holds rows as compressed bits.

  @param colIdx is the column index.
  
  @return void, with output reference bit matrix.
*/
void BitMatrix::SetColumn(const BV *vec, int colIdx) {
  unsigned int slotBits = BV::SlotBits();
  int slotRow = 0;
  unsigned int slot = 0;
  for (unsigned int baseRow = 0; baseRow < nRow; baseRow += slotBits, slot++) {
    unsigned int sourceSlot = vec->Slot(slot);
    unsigned int mask = 1;
    unsigned int supRow = nRow < baseRow + slotBits ? nRow : baseRow + slotBits;
    for (unsigned int row = baseRow; row < supRow; row++, mask <<= 1) {
      if (sourceSlot & mask) { // row is in-bag.
	SetBit(row, colIdx);
      }
    }
    slotRow += slotBits;
  }
}


/**
   @brief Static entry.  Exports matrix as vector of vectors.

   @param outBag outputs the columns as vectors.

   @return void, with output vector parameter.
 */
void BitMatrix::Export(const std::vector<unsigned int> &_raw, unsigned int _nRow, std::vector<std::vector<unsigned int> > &vecOut) {
  unsigned int _nCol = vecOut.size();
  BitMatrix *bm = new BitMatrix(_nRow, _nCol, _raw);
  bm->Export(_nRow, vecOut);

  delete bm;
}


/**
   @brief Exports matrix as vector of column vectors.

   @param _nRow is the external row count.

   @return void, with output reference parameter.
 */
void BitMatrix::Export(unsigned int _nRow, std::vector<std::vector<unsigned int> > &outCol) {
  for (unsigned int i = 0; i < stride; i++) {
    outCol[i] = std::vector<unsigned>(_nRow);
    ColExport(_nRow, outCol[i], i);
  }
}


/**
   @brief Exports an individual column to a uint vector.

   @param _nRow is the external row count.

   @param outCol outputs the column.

   @param colIdx is the column index.

   @return void, with output reference vector.
 */
void BitMatrix::ColExport(unsigned int _nRow, std::vector<unsigned int> &outCol, unsigned int colIdx) {
  for (unsigned int row = 0; row < _nRow; row++)
    outCol[row] = IsSet(row, colIdx) ? 1 : 0;
}
