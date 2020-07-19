// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bheap.h

   @brief Homemade priority queue.

   @author Mark Seligman
 */

#ifndef CORE_BHEAP_H
#define CORE_BHEAP_H


#include "typeparam.h"

#include <vector>

using namespace std;

/**
   @brief Ad hoc container for simple priority queue.
 */
struct BHPair {
  double key;  // TODO:  templatize.
  PredictorT slot; // Slot index.
};


/**
   @brief Implementation of binary heap tailored to RunAccums.

   Not so much a class as a collection of static methods.
*/
struct BHeap {
  /**
     @brief Determines index of parent.
   */
  static inline int parent(int idx) { 
    return (idx-1) >> 1;
  };


  /**
     @brief Empties the queue.

     @param pairVec are the queue records.

     @param[out] lhOut outputs the popped slots, in increasing order.

     @param pop is the number of elements to pop.  Caller enforces value > 0.
  */
  static void depopulate(BHPair pairVec[],
                         unsigned int lhOut[],
                         unsigned int pop);

  /**
     @brief Inserts a key, value pair into the heap at next vacant slot.

     Heap updates to move element with maximal key to the top.

     @param pairVec are the queue records.

     @param slot_ is the slot position.

     @param key_ is the associated key.
  */
  static void insert(BHPair pairVec[],
                     unsigned int slot_,
                     double key_);

  /**
     @brief Pops value at bottom of heap.

     @param pairVec are the queue records.

     @param bot indexes the current bottom.

     @return popped value.
  */
  static unsigned int slotPop(BHPair pairVec[],
                              int bot);

  /**
     @brief Permutes a zero-based set of contiguous values.

     @param nSlot is the number of values.

     @return vector of permuted indices.
   */
  static vector<size_t> permute(IndexT nSlot);
};

#endif
