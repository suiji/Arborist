// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file restage.h

   @brief Class definitions supporting maintenance of per-predictor sample orderings.

   @author Mark Seligman

 */


#ifndef ARBORIST_RESTAGE_H
#define ARBORIST_RESTAGE_H

class MapNode {
  int splitIdx; // Position in map.
  int lNext;
  int rNext;
  int lhIdxCount;
  int rhIdxCount;
  int startIdx;
  int endIdx;

 public:
  void UpdateIndices(int &lhIdx, int &rhIdx);

  /**
     @brief Accessor.

     @return ending index of map node.
   */
  inline int EndIdx() {
    return endIdx;
  }

  inline int LNext() {
    return lNext;
  }

  inline int RNext() {
    return rNext;
  }


  /**
     @brief Initializes all fields essential for restaging.
   */
  void Init(int _splitIdx, int _lNext, int _rNext, int _lhIdxCount, int _rhIdxCount, int _startIdx, int _endIdx) {
    splitIdx = _splitIdx;
    lNext = _lNext; // Negative indices denote terminal subnodes.
    rNext = _rNext; // " "
    lhIdxCount = _lhIdxCount;
    rhIdxCount = _rhIdxCount;
    startIdx = _startIdx;
    endIdx = _endIdx;
  }

  /**
     @param bv is the bit vector to check.

     @param sIdx is the sample index at which to test.

     @return true iff sample index is set in the bit vector.
   */
  inline bool IsSet(const unsigned int bv[], unsigned int sIdx) {
    const unsigned int slotBits = 8 * sizeof(unsigned int);
    unsigned int slot = sIdx / slotBits; // Compiler should generate right-shift.
    unsigned int mask = 1 << (sIdx - (slot * slotBits));
    return (bv[slot] & mask) != 0;
  }

  void Restage(const SPNode source[], const unsigned int sIdxSource[], SPNode targ[], unsigned int sIdxTarg[], const unsigned int sIdxLH[], const unsigned int sIdxRH[], int lhIdx, int rhIdx);
  void RestageLR(const class SPNode *source, const unsigned int sIdxSource[], class SPNode *targ, unsigned int sIdxTarg[], int startIdx, int endIdx, const unsigned int bvL[], int lhIdx, int rhIdx);
  void RestageSingle(const class SPNode *source, const unsigned int sIdxSource[], class SPNode *targ, int unsigned sIdxTarg[], int startIdx, int endIdx, const unsigned int bv[], int idx);

  void Singletons(class SplitPred *splitPred, const class SPNode targ[], int predIdx, int lhIdx, int rhIdx);
};

class RestageMap {
  int splitPrev; // Number of splits in the level just concluded.
  int splitNext; // " " next level.
  MapNode *mapNode;

 protected:
  class SplitPred *splitPred;
  unsigned int bitSlots; // Useful for coprocessor alignment of predicates.
  unsigned int *sIdxLH; // Predicate for live LH indices.
  unsigned int *sIdxRH; // Predicate for live RH indices.
  int endPrev; // Terminus of live indices in previous level.
  int endThis; // Terminus of live indices in this level.
  int rhIdxNext; // Starting index of next level RH:   stable partition.
  
 public:
  static int nPred;
  static int nSamp; // Useful to coprocessor.
  static void Immutables(int _nPred, int _nSamp);
  static void DeImmutables();
  RestageMap(class SplitPred *_splitPred, unsigned int _bagCount, int _splitPrev, int _splitNext);
  ~RestageMap();
  void RestageLevel(class SamplePred *samplePred, int level);
  void RestagePred(const class SPNode source[], const unsigned int sIdxSource[], class SPNode targ[], unsigned int sIdxTarg[], int predIdx) const;
  void ConsumeSplit(int _splitIdx, int _lNext, int _rNext, int _lhIdxCount, int _rhIdxCount, int _startIdx, int _endIdx);
  void Conclude(const class Index *index);//, int _splitPrev);
};

#endif

