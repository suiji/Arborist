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


void BV::resize(size_t bitMin) {
  size_t slotMin = slotAlign(bitMin);
  if (nSlot >= slotMin)
    return;

  size_t slotsNext = nSlot;
  while (slotsNext < slotMin)
    slotsNext <<= 1;
  rawV.resize(slotsNext);
  raw = &rawV[0];
  nSlot = slotsNext;
}


void BV::delEncode(const vector<IndexT>& delPos) {
  const unsigned int slotBits = getSlotElts();
  unsigned int log2Bits = 0ul;
  while (slotBits > (1ul << log2Bits))
    log2Bits++;

  IndexT pos = 0;
  IndexT slotPrev = 0;
  unsigned int bits = 0ul;
  for (IndexT sIdx = 0; sIdx < delPos.size(); sIdx++) {
    pos += delPos[sIdx];
    IndexT slot = pos >> log2Bits;
    if (slot != slotPrev) {
      setSlot(slotPrev, bits);
      bits = 0ul;
    }
    bits |= (1ul << (pos & (slotBits - 1)));
    slotPrev = slot;
  }
  setSlot(slotPrev, bits); // Flushes remaining bits.
}


BitMatrix::BitMatrix(size_t nRow_, unsigned int nCol_) :
  BV(nRow_ * Stride(nCol_)),
  nRow(nRow_),
  stride(Stride(nCol_)) {
}


BitMatrix::BitMatrix(RawT raw_[], size_t nRow_, size_t nCol_) :
  BV(raw_, nRow_ * Stride(nCol_)),
  nRow(nRow_),
  stride(nRow > 0 ? Stride(nCol_) : 0) {
}


BitMatrix::~BitMatrix() {
}


void BitMatrix::dump(size_t nRow_, vector<vector<unsigned int> > &outCol) const {
  for (size_t i = 0; i < stride; i++) {
    outCol[i] = vector<unsigned int>(nRow_);
    colDump(nRow_, outCol[i], i);
  }
}

void BitMatrix::colDump(size_t _nRow, vector<unsigned int> &outCol, unsigned int colIdx) const {
  for (size_t row = 0; row < _nRow; row++)
    outCol[row] = testBit(row, colIdx) ? 1 : 0;
}


BVJagged::BVJagged(RawT raw_[],
		   const vector<size_t>& rowExtent_) :
  BV(raw_, rowExtent_.back()),
  rowHeight(move(rowExtent_)),
  nRow(rowExtent_.size()) {
}


BVJagged::~BVJagged() {
}


vector<vector<RawT>> BVJagged::dump() const {
  vector<vector<RawT>> outVec(nRow);
  for (IndexT row = 0; row < nRow; row++) {
    outVec[row] = rowDumpRaw(row);
  }
  return outVec;
}


/**
   @brief Exports contents for an individual row.
 */
vector<RawT> BVJagged::rowDumpRaw(size_t rowIdx) const {
  unsigned int base = rowIdx == 0 ? 0 : rowHeight[rowIdx-1];
  unsigned int extent = rowHeight[rowIdx] - base;
  return dumpVec(base, extent);
}

