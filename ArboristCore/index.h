// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file index.h

   @brief Definitions for classes maintaining the index-tree representation.

   @author Mark Seligman

 */

#ifndef ARBORIST_INDEX_H
#define ARBORIST_INDEX_H

/**
   Index tree node fields associated with the response, viz., invariant across
   predictors.  IndexNodes of the index tree can be thought of as representing
   collections of sample indices. The two subnodes of a node, moreover, can
   be thought of as defining a bipartition of the parent's index collection.


   IndexNodes only live within a single level, with fields being reused as
   new levels are seen.
*/
class IndexNode {
  static int totLevels;
  static void SplitOffsets(int splitCount);
  static bool CheckStorage(int splitCount, int splitNext, int leafNext);
  static void LateFields(int splitCount);
 protected:
  static IndexNode *indexNode; // Vector of splits for one level:  reallocatable.
  static int levelMax; // High-watermark for allocation and re-allocation.
  static void NextLevel(int _splitIdx, int _ptId, int _idxCount, int _sCount, double _sum, double _minInfo);
 public: // The three integer values are all non-negative.
  int lhStart; // Start index of LHS data in buffer.
  int idxCount; // # distinct indices in the node.
  int sCount;  // # samples subsumed by this node.
  double sum; // Sum of all responses in node.
  double minInfo; // Minimum acceptable information on which to split.
  double preBias; // Inf of information values eligible for splitting.
  int ptId; // Index of associated PTSerial node.

  static void TreeInit(int _levelMax, int _bagCount, int _nSamp, double _sum);
  static void TreeClear();
  static int Levels();

  static void Factory(int _minHeight, int _totLevels);
  static void ReFactory();
  static void DeFactory();

  /**
     @brief Returns the pretree index associated with an index node.

     @param splitIdx is the split index referenced.

     @return pretree index.
   */
  static inline int PTId(int splitIdx) {
    return indexNode[splitIdx].ptId;
  }

  /**
     @brief Returns the pre-bias of an index node.

     @param splitIdx is the split index referenced.

     @return pre-bias value.
   */
  static inline double GetPrebias(int splitIdx) {
    return indexNode[splitIdx].preBias;
  }

  /**
     @brief Exposes fields relevant for SplitPred methods.   N.B.:  Not all methods use all fields.

     @param _lhStart outputs the left-most index.

     @param _end outputs the right-most index.

     @param _sCount outputs the total sample count.

     @param _sum outputs the sum.

     @param _preBias outputs the pre-bias.

     @return void, with output reference parameters.
  */
  static void inline SplitFields(int splitIdx, int &_lhStart, int &_end, int &_sCount, double &_sum, double &_preBias) {
    IndexNode *idxNode = &indexNode[splitIdx];
    _lhStart = idxNode->lhStart;
    _end = _lhStart + idxNode->idxCount - 1;
    _sCount = idxNode->sCount;
    _sum = idxNode->sum;
    _preBias = idxNode->preBias;
  }

  /**
     @brief Exposes fields needed for computing pre-bias.

     @param splitIdx is the split index referenced.

     @param _sum outputs the sum of response values.

     @param _sCount outputs the count of samples.

     @return void, with output reference parameters.
  */
  static void inline PrebiasFields(int splitIdx, double &_sum, int &_sCount) {
    IndexNode *idxNode = &indexNode[splitIdx];
    _sum = idxNode->sum;
    _sCount = idxNode->sCount;
  }

  /**
   @brief Sets pre-bias field.  Called from Response methods.

   @param splitIdx is the split index referenced.

   @param _preBias is the pre-bias value to set.

   @return void, with field-valued side effect.
  */
  static void inline SetPrebias(int splitIdx, double _preBias) {
    indexNode[splitIdx].preBias = _preBias; 
  }    
};

/**
   @brief Caches intermediate IndexNode contents during intra-level transfer.
*/
class NodeCache : public IndexNode {
  static NodeCache *nodeCache; // ReFactoryable.
  static int minHeight;
  void Consume(int lhSplitNext, int &lhCount, int &rhCount);
  void Splitable(int level);
  int SplitCensus(int &lhSplitNext, int &rhSplitNext, int &leafNext);
  static void ReFactory();
 public:
  static void Factory(int _minHeight);
  static void DeFactory();
  class SplitSig *splitSig; // Convenient to cache for LH/RH partition.
  int ptL; // LH index into pre-tree:  splits only.
  int ptR; // RH index into pre-tree:  splits only.
  static void CacheNodes(int splitCount);
  static int InterLevel(int level, int splitCount, int &lhSplitNext, int &rhSplitNext, int &leafNext);
  static void NextLevel(int splitCount, int lhSplitNext, int totLhIdx, bool reFac);
  static void TreeInit();
  static void TreeClear();

  /**
     @brief Copies indexNode entry into corresponding nodeCache.

     @param splitIdx is the index at which to copy.

     @return void.
   */
  static inline void Cache(int splitIdx) {
    IndexNode *nd = &indexNode[splitIdx];
    NodeCache *nc = &nodeCache[splitIdx];

    nc->lhStart = nd->lhStart;
    nc->idxCount = nd->idxCount;
    nc->sCount = nd->sCount;
    nc->sum = nd->sum;
    nc->preBias = nd->preBias;
    nc->ptId = nd->ptId;
    nc->minInfo = nd->minInfo;
    nc->ptL = nc->ptR = -1; // Terminal until shown otherwise.
  }

  /**
     @brief Exposes field values relevant for restating

     @param splitIdx is the reference index.

     @param _ptL outputs the pretree index of the left subnode.

     @param _ptR outputs the pretree index of the right subnode.

     @param _lhStart outputs the position of the left-most index.

     @param _endIdx outputs the position of the right-most index.

     @return void, with reference parameters.
   */
  // Returns those fields of the cached node which are relevant for restating and
  // replaying:  the PreTree indices of the left and right offspring, as well
  // as the start and end positions of the left-hand bipartition.
  //
  static inline void RestageFields(int splitIdx, int &_ptL, int &_ptR, int &_lhStart, int &_endIdx) {
    NodeCache *ndc = &nodeCache[splitIdx];
    _ptL = ndc->ptL;
    _ptR = ndc->ptR;
    _lhStart = ndc->lhStart;
    _endIdx = _lhStart + ndc->idxCount - 1;
  }

  static double ReplayNum(int splitIdx, int predIdx, int level, int lhIdxCount);

  /**
     @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next

     @param _SCount is the count of samples subsumed by the node.

     @param _idxCount is the count of indices subsumed by the node.

     @return true iff the node subsumes too few samples or is representable as a
     single buffer element.
  */
  static inline bool TerminalSize(int _SCount, int _idxCount) {
    return (_SCount < minHeight) || (_idxCount <= 1);
  }
};

#endif
