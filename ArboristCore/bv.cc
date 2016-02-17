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
BV::BV(unsigned int len, bool slotWise) : nSlot(slotWise ? len : SlotAlign(len)) {
  raw = new unsigned int[nSlot];
  for (unsigned int i = 0; i < nSlot; i++) {
    raw[i] = 0;
  }
}


/**
 */
BV::BV(const std::vector<unsigned int> &_raw) : nSlot(_raw.size()) {
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
void BV::Consume(std::vector<unsigned int> &out, unsigned int bitEnd) {
  unsigned int slots = bitEnd == 0 ? nSlot : SlotAlign(bitEnd);
  out.reserve(slots);
  out.insert(out.end(), raw, raw + slots);
}


/**
 */
BitMatrix::BitMatrix(unsigned int _nRow, unsigned int _nCol) : BV(_nRow * _nCol), stride(_nCol) {
}


/**
   @brief Constructor.  Sets stride to zero if empty.
 */
BitMatrix::BitMatrix(unsigned int _nRow, unsigned int _nCol, const std::vector<unsigned int> &_raw) : BV(_raw), stride(_raw.size() > 0 ? _nCol : 0) {
}


/**
 */
BitMatrix::~BitMatrix() {
}


/**
 */
BVJagged::BVJagged(const std::vector<unsigned int> &_raw, const std::vector<unsigned int> _offset) : BV(_raw) {
  unsigned int rowCount = _offset.size();
  offset = new unsigned int[rowCount];
  for (unsigned int i = 0; i < rowCount; i++) {
    offset[i] = _offset[i];
  }
}


/**
 */
BVJagged::~BVJagged() {
  delete [] offset;
}
