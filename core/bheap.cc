// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bheap.cc

   @brief Methods for maintaining simple binary heap.

   @author Mark Seligman
 */


#include "bheap.h"
#include "callback.h"

unsigned int BHeap::slotPop(BHPair pairVec[], int bot) {
  unsigned int ret = pairVec[0].slot;
  if (bot == 0)
    return ret;
  
  // Places bottom element at head and refiles.
  unsigned int idx = 0;
  int slotRefile = pairVec[idx].slot = pairVec[bot].slot;
  double keyRefile = pairVec[idx].key = pairVec[bot].key;
  int descL = 1;
  int descR = 2;

    // 'descR' remains the lower of the two descendant indices.
    //  Some short-circuiting below.
    //
  while((descR <= bot && keyRefile > pairVec[descR].key) || (descL <= bot && keyRefile > pairVec[descL].key)) {
    int chIdx =  (descR <= bot && pairVec[descR].key < pairVec[descL].key) ?  descR : descL;
    pairVec[idx].key = pairVec[chIdx].key;
    pairVec[idx].slot = pairVec[chIdx].slot;
    pairVec[chIdx].key = keyRefile;
    pairVec[chIdx].slot = slotRefile;
    idx = chIdx;
    descL = 1 + (idx << 1);
    descR = (1 + idx) << 1;
  }

  return ret;
}


void BHeap::depopulate(BHPair pairVec[], PredictorT idxRank[], PredictorT pop) {
  for (int bot = pop - 1; bot >= 0; bot--) {
    idxRank[slotPop(pairVec, bot)] = pop - (1 + bot);
  }
}


void BHeap::insert(BHPair pairVec[], unsigned int slot_, double key_) {
  unsigned int idx = slot_;
  BHPair input;
  input.key = key_;
  input.slot = slot_;
  pairVec[idx] = input;

  int parIdx = parent(idx);
  while (parIdx >= 0 && pairVec[parIdx].key > key_) {
    pairVec[idx] = pairVec[parIdx];
    pairVec[parIdx] = input;
    idx = parIdx;
    parIdx = parent(idx);
  }
}


vector<size_t> BHeap::permute(IndexT nSlot) {
  auto vUnif = CallBack::rUnif(nSlot);
  vector<BHPair> heap(nSlot);
  for (IndexT slot = 0; slot < nSlot; slot++) {
    BHeap::insert(&heap[0], slot, vUnif[slot]);
  }

  IndexT i = 0;
  vector<size_t> shuffle(nSlot);
  for (IndexT heapSize = nSlot; heapSize > 0; heapSize--) {
    shuffle[i++] = BHeap::slotPop(&heap[0], heapSize - 1);
  }

  return shuffle;
}
