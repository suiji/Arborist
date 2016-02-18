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
  int idxNextL;  // Starting left index
  int idxNextR; // Starting right index.
  int startIdx;  // Start index of predecessor.
  int endIdx;    // End index of predecessor.

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

  void Restage(const class SPNode source[], const unsigned int sIdxSource[], class SPNode targ[], unsigned int sIdxTarg[], const BV *sIdxLH, const BV *sIdxRH);
  void RestageLR(const class SPNode *source, const unsigned int sIdxSource[], class SPNode *targ, unsigned int sIdxTarg[], int startIdx, int endIdx, const class BV *bvL, int lhIdx, int rhIdx);
  void RestageSingle(const class SPNode *source, const unsigned int sIdxSource[], class SPNode *targ, int unsigned sIdxTarg[], int startIdx, int endIdx, const class BV *bv, int idx);

  void Singletons(class SplitPred *splitPred, const class SPNode targ[], int predIdx);
};


class RestageMap {
  static int nPred;
  int splitPrev; // Number of splits in the level just concluded.
  int splitNext; // " " next level.
  MapNode *mapNode;

 protected:
  class SplitPred *splitPred;
  class BV *sIdxLH; // Predicate for live LH indices.
  class BV *sIdxRH; // Predicate for live RH indices.
  int endPrev; // Terminus of live indices in previous level.
  int endThis; // Terminus of live indices in this level.
  int rhIdxNext; // Starting index of next level RH:   stable partition.
  
 public:
  RestageMap(class SplitPred *_splitPred, unsigned int _bagCount, int _splitPrev, int _splitNext);
  ~RestageMap();
  static void Immutables(int _nPred);
  static void DeImmutables();

  void RestageLevel(class SamplePred *samplePred, unsigned int level);
  void ConsumeSplit(int _splitIdx, int _lNext, int _rNext, int _lhIdxCount, int _rhIdxCount, int _startIdx, int _endIdx);
  void Conclude(const class Index *index);//, int _splitPrev);
};

#endif

