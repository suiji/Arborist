// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_INDEX_H
#define ARBORIST_INDEX_H

// Index tree node fields associated with the response, viz., invariant across
// predictors.  IndexNodes of the index tree can be thought of as representing
// collections of sample indices. The two subnodes of a node, moreover, can
// be thought of as defining a bipartition of the parent's index collection.
//
// IndexNodes only live within a single level, with fields being reused as
// new levels are seen.
//
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

  static inline int PTId(int splitIdx) {
    return indexNode[splitIdx].ptId;
  }

  static inline double GetPrebias(int splitIdx) {
    return indexNode[splitIdx].preBias;
  }

  // Fills in fields relevant for SplitPred methods.
  // N.B.:  Not all methods use all fields.
  //
  static void inline SplitFields(int splitIdx, int &_lhStart, int &_end, int &_sCount, double &_sum, double &_preBias) {
    IndexNode *idxNode = &indexNode[splitIdx];
    _lhStart = idxNode->lhStart;
    _end = _lhStart + idxNode->idxCount - 1;
    _sCount = idxNode->sCount;
    _sum = idxNode->sum;
    _preBias = idxNode->preBias;
  }

  // Fills in those fields needed for computing pre-bias.
  //
  static void inline PrebiasFields(int splitIdx, double &_sum, int &_sCount) {
    IndexNode *idxNode = &indexNode[splitIdx];
    _sum = idxNode->sum;
    _sCount = idxNode->sCount;
  }

  // Called from Response methods.
  //
  static void inline SetPrebias(int splitIdx, double _preBias) {
    indexNode[splitIdx].preBias = _preBias; 
  }    
};

// Caches intermediate IndexNode contents during intra-level transfer.
//
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
  }

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

  // Invoked from the RHS or LHS of a split to determine whether the node persists to the next
  // level.  Returns true if the node subsumes too few samples or is representable as a
  // single buffer element.
  //
  static inline bool TerminalSize(int _SCount, int _idxCount) {
    return (_SCount < minHeight) || (_idxCount <= 1);
  }
};

#endif
