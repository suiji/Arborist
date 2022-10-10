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

const size_t BV::full = size_t(1ull);
const size_t BV::allOnes = size_t(~0ull);
const size_t BV::slotSize = sizeof(BVSlotT);
const size_t BV::slotElts = 8 * slotSize;


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


BitMatrix::BitMatrix(unsigned int nRow_,
		     IndexT nCol_) :
  BV(nRow_ * Stride(nCol_)),
  nRow(nRow_),
  stride(Stride(nCol_)) {
}


BitMatrix::BitMatrix(const BVSlotT raw_[],
		     unsigned int nRow_,
		     IndexT nCol_) :
  BV(raw_, nRow_ * Stride(nCol_)),
  nRow(nRow_),
  stride(nRow > 0 ? Stride(nCol_) : 0) {
}


BitMatrix::~BitMatrix() {
}


void BitMatrix::dump(unsigned int nRow_,
		     vector<vector<BVSlotT> > &outCol) const {
  for (size_t i = 0; i < stride; i++) {
    outCol[i] = vector<BVSlotT>(nRow_);
    colDump(nRow_, outCol[i], i);
  }
}

void BitMatrix::colDump(unsigned int nRow_,
			vector<BVSlotT>& outCol,
			IndexT colIdx) const {
  for (unsigned int row = 0; row < nRow_; row++)
    outCol[row] = testBit(row, colIdx) ? 1 : 0;
}


BVJagged::BVJagged(const BVSlotT raw_[],
		   const vector<size_t>& rowExtent_) :
  BV(raw_, rowExtent_.back()),
  rowHeight(std::move(rowExtent_)),
  nRow(rowExtent_.size()) {
}


BVJagged::~BVJagged() {
}


vector<vector<BVSlotT>> BVJagged::dump() const {
  vector<vector<BVSlotT>> outVec(nRow);
  for (IndexT row = 0; row < nRow; row++) {
    outVec[row] = rowDumpRaw(row);
  }
  return outVec;
}


/**
   @brief Exports contents for an individual row.
 */
vector<BVSlotT> BVJagged::rowDumpRaw(size_t rowIdx) const {
  unsigned int base = rowIdx == 0 ? 0 : rowHeight[rowIdx-1];
  unsigned int extent = rowHeight[rowIdx] - base;
  return dumpVec(base, extent);
}

