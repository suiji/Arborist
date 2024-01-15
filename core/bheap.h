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

#include <cmath>
#include <vector>

using namespace std;

/**
   @brief Ad hoc container for simple priority queue.
 */
template<typename slotType>
struct BHPair {
  double key;
  slotType slot; // Slot index.

  BHPair(double key_,
	 slotType slot_) :
    key(key_),
    slot(slot_) {
  }

  BHPair() = default;
};


/**
   @brief Maintains partial sorting of a vector of pairs.
 */
namespace PQueue {

  /**
     @brief Adjust partial ordering for addition of element.
   */
  template<typename slotType>
  void insert(BHPair<slotType> pairVec[],
	      slotType tail) {
    const BHPair<slotType> input = pairVec[tail];
    slotType idx = tail;
    while (idx > 0) {
      slotType parIdx = (idx - 1) >> 1;
      if (pairVec[parIdx].key <= input.key)
	break;
      pairVec[idx] = pairVec[parIdx];
      pairVec[parIdx] = input;
      idx = parIdx;
    }
  }

  
  /**
     @brief Adjusts partial ordering for removal of element.
   */
  template<typename slotType>
  void refile(BHPair<slotType> bhPair[],
	      slotType tail) {
    // Places tail element at head and refiles.
    const slotType slotRefile = bhPair[0].slot = bhPair[tail].slot;
    const double keyRefile = bhPair[0].key = bhPair[tail].key;

  // 'descR' remains the lower of the two descendant indices.
  //  Some short-circuiting below.
  //
    slotType idx = 0;
    slotType descL = 1;
    slotType descR = 2;
    while((descR <= tail && keyRefile > bhPair[descR].key) || (descL <= tail && keyRefile > bhPair[descL].key)) {
      slotType chIdx =  (descR <= tail && bhPair[descR].key < bhPair[descL].key) ?  descR : descL;
      bhPair[idx].key = bhPair[chIdx].key;
      bhPair[idx].slot = bhPair[chIdx].slot;
      bhPair[chIdx].key = keyRefile;
      bhPair[chIdx].slot = slotRefile;
      idx = chIdx;
      descL = 1 + (idx << 1);
      descR = (1 + idx) << 1;
    }
  }


  /**
     @brief Empties the queue.

     @param pairVec are the queue records.

     @param nElt is the number of elements to pop:  > 0.
  */
  template<typename slotType>
  vector<slotType> depopulate(BHPair<slotType> pairVec[],
			      slotType nElt) {
    vector<slotType> idxRank(nElt);
    for (slotType pairIdx = 0; pairIdx < nElt; pairIdx++) {
      idxRank[pairVec[0].slot] = pairIdx;
      refile<slotType>(pairVec, nElt - (pairIdx + 1));
    }
    return idxRank;
  }


  /**
     @brief Inserts a key, value pair into the queue.

     @param pairVec are the queue records.

     @param key is the associated key.

     @param slot is the slot position.
  */
  template<typename slotType>
  void insert(BHPair<slotType> pairVec[], double key, slotType slot) {
    pairVec[slot] = BHPair<slotType>(key, slot);
    insert<slotType>(pairVec, slot);
  }
}


/**
   @brief Internal implementation of binary heap.
*/
template<typename slotType>
struct BHeap {
  vector<BHPair<slotType>> bhPair;

public:

  size_t size() const {
    return bhPair.size();
  }  


  bool empty() const {
    return bhPair.size() == 0;
  }

  
  /**
     @brief Removes a single item from the head of the queue.

     @return slot value.
   */
  slotType pop() {
    slotType slot = bhPair.front().slot;
    PQueue::refile<slotType>(&bhPair[0], bhPair.size() - 1);
    bhPair.pop_back();
    return slot;
  }
  

  /**
     @brief Removes items from the queue.

     @param nElt is the number of items to remove.

     @return ranks of popped items.
  */
  vector<slotType> depopulate(size_t nElt = 0) {
    vector<slotType> idxRank(nElt == 0 ? bhPair.size() : min(bhPair.size(), nElt));
    for (slotType pairIdx = 0; pairIdx < idxRank.size(); pairIdx++) {
      idxRank[pop()] = pairIdx;
    }
    return idxRank;
  }


  /**
     @brief Inserts a key, value pair into the heap at next vacant slot.

     Heap updates to move element with maximal key to the top.

     @param key is the associated key.
  */
  void insert(double key) {
    bhPair.emplace_back(key, bhPair.size());
    PQueue::insert<slotType>(&bhPair[0], bhPair.back().slot);
  }
};

#endif
