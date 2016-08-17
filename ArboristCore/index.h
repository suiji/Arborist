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

#include <vector>

/**
   Index tree node fields associated with the response, viz., invariant across
   predictors.  IndexNodes of the index tree can be thought of as representing
   collections of sample indices. The two subnodes of a node, moreover, can
   be thought of as defining a bipartition of the parent's index collection.

   IndexNodes only live within a single level, with fields being reused as
   new levels are seen.
*/
class IndexNode {
  double preBias; // Inf of information values eligible for splitting.
 public: // The three integer values are all non-negative.
  IndexNode();
  unsigned int splitIdx; // Position within containing vector:  split index.
  unsigned int lhStart; // Start index of LHS data in buffer.
  unsigned int idxCount; // # distinct indices in the node.
  unsigned int sCount;  // # samples subsumed by this node.
  double sum; // Sum of all responses in node.
  double minInfo; // Minimum acceptable information on which to split.
  unsigned int ptId; // Index of associated PTSerial node.
  unsigned char path; // Bitwise record of recent reaching L/R path.

  double PrebiasReg();
  double PrebiasCtg(const double sumSquares[]);

  /**
     @brief Outputs fields used by pre-bias computation.

     @param _sCount outputs the sample count.

     @param _sum outputs the sum

     @return void.
   */
  inline void PrebiasFields(unsigned int &_sCount, double &_sum) {
    _sCount = sCount;
    _sum = sum;
  }

  
  /**
     @brief Accessor for 'preBias' field.

     @return reference to 'preBias' field.
  */
  inline double &Prebias() {
    return preBias;
  }


  /**
     @brief Sets nearly all (invariant) fields for the upcoming split methods.
     The only exceptions are the "late" values, 'preBias' and 'lhStart'.
     @see LateFields

     @param _splitIdx is the index within the containing vector.

     @param _ptId is the pretree node index.

     @param _idxCount is the count indices represented.

     @param _sCount is the count of samples represented.

     @param _sum is the sum of response values at the indices represented.

     @param _minInfo is the minimal information content suitable to split either child.

     @return void.
  */
  void Init(int _splitIdx, unsigned int _start, unsigned int _ptId, int _idxCount, unsigned int _sCount, double _sum, double _minInfo, unsigned char _path) {
    splitIdx = _splitIdx;
    ptId = _ptId;
    lhStart = _start;
    idxCount = _idxCount;
    sCount = _sCount;
    sum = _sum;
    minInfo = _minInfo;
    path = _path;
  }

  inline void PathCoords(unsigned int &_start, unsigned int &_extent) {
    _start = lhStart;
    _extent = idxCount;
  }
  
  /**
     @return reference to 'lhStart' field.
   */
  inline unsigned int &Start() {
    return lhStart;
  }

  inline unsigned int &IdxCount() {
    return idxCount;
  }
  
  /**
     @brief Exposes fields relevant for SplitPred methods.   N.B.:  Not all methods use all fields.

     @param _lhStart outputs the left-most index.

     @param _end outputs the right-most index.

     @param _sCount outputs the total sample count.

     @param _sum outputs the sum.

     @return preBias, with output parameters.
  */
  double inline SplitFields(unsigned int &_lhStart, unsigned int &_end, unsigned int &_sCount, double &_sum) const {
    _lhStart = lhStart;
    _end = _lhStart + idxCount - 1;
    _sCount = sCount;
    _sum = sum;
    return preBias;
  }

  void inline Extent(unsigned int &_lhStart, unsigned int &_end) const {
    _lhStart = lhStart;
    _end = lhStart + idxCount - 1;
  }


  /**
     @brief Exposes minimum-information value for the node.

     @return minInfo value.
   */
  inline double MinInfo() const {
    return minInfo;
  }
};


/**
   @brief Caches intermediate IndexNode contents during intra-level transfer.
*/
class NodeCache : public IndexNode {
  class SSNode *ssNode; // Convenient to cache for LH/RH partition.
  bool terminal; // True unless next-level descendants produced.
  static unsigned int minNode;
  unsigned int lhIdxCount; // Total indices over LH:  splits only.
  unsigned int lhSCount; // Total samples cover LH:  splits only.
  double lhSum; // Sum of responses over LH:  splits only.
  unsigned int ptL; // LH index into pre-tree:  splits only.
  unsigned int ptR; // RH index into pre-tree:  splits only.
 public:
  static void Immutables(unsigned int _minNode);
  static void DeImmutables();
  NodeCache();
  void Consume(class PreTree *preTree, class SamplePred *samplePred, class Bottom *bottom);
  void Successors(class Index *index, class PreTree *preTree, class SamplePred *samplePred, class Bottom *bottom, unsigned int lhSplitNext, unsigned int &lhCount, unsigned int &rhCount);
  void SplitCensus(unsigned int &lhSplitNext, unsigned int &rhSplitNext, unsigned int &leafNext);

  /**
     @brief Copies indexNode entry into corresponding nodeCache.

     @param nd is the IndexNode to copy.

     @return void.
   */
  inline void Cache(IndexNode *nd, class SSNode *argMax) {
    splitIdx = nd->splitIdx;
    lhStart = nd->lhStart;
    idxCount = nd->idxCount;
    sCount = nd->sCount;
    sum = nd->sum;
    ptId = nd->ptId;
    minInfo = nd->minInfo;
    path = nd->path,
    ptL = ptR = 0; // Terminal until shown otherwise.
    ssNode = argMax;
  }
  

  /**
    @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next level.
    
    MUST guarantee that no zero-length "splits" have been introduced.
    Not only are these nonsensical, but they are also dangerous, as they violate
    various assumptions about the integrity of the intermediate respresentation.

    @param _idxCount is the count of indices subsumed by the node.

    @return true iff the node subsumes more than minimal count of buffer elements.
  */
  inline bool Splitable(unsigned int _idxCount) const {
    return _idxCount > minNode;
  }


  
  /**
     @brief Transmits location coordinates for interlevel activity.

     @param _start outputs starting index position.

     @param _extent output index count.

     @return true iff no successors produced.
   */
  inline bool Terminal(unsigned int &_start, unsigned int &_extent, unsigned int &_lhIdxCount) const {
    _start = lhStart;
    _extent = idxCount;
    _lhIdxCount = lhIdxCount; // nonzero iff nonterminal.

    return terminal;
  }
  
};

class Index {
  static unsigned int totLevels;
  NodeCache *CacheNodes(const std::vector<class SSNode*> &argMax);
  void ArgMax(NodeCache nodeCache[]);
  unsigned int LevelCensus(NodeCache nodeCache[], unsigned int levelCount, unsigned int &lhSplitNext, unsigned int &leafNext);
  NodeCache *LevelConsume(unsigned int levelCount, unsigned int &splitNext, unsigned int &lhSplitNext, unsigned int &leafNext);
  void LevelProduce(NodeCache *nodeCache, unsigned int level, unsigned int levelCount, unsigned int splitNext, unsigned int lhSplitNext, unsigned int leafNext);
 protected:
  IndexNode *indexNode;  
  const unsigned int bagCount;
  unsigned int levelBase; // Pre-tree index at which level's nodes begin.
  unsigned int levelWidth; // Count of pretree nodes at frontier.
  bool *ntLH;
  bool *ntRH;
  static class PreTree *OneTree(class SamplePred *_samplePred, class Bottom *_bottom, int _nSamp, int _bagCount, double _bagSum);

 public:
  static void Immutables(unsigned int _minNode, unsigned int _totLevels);
  static void DeImmutables();
  class SamplePred *samplePred;
  class PreTree *preTree;
  class Bottom *bottom;
  Index(class SamplePred *_samplePred, class PreTree *_preTree, class Bottom *_bottom, int _nSamp, int _bagCount, double _sum);
  ~Index();

  static class PreTree **BlockTrees(class Sample **sampleBlock, int _treeBlock);
  void SetPrebias();
  void Levels();
  void PredicateBits(class BV *bitsLH, class BV *bitsRH, int &lhIdxTot, int &rhIdxTot) const;
  void LRLive(class Bottom *bottom, const NodeCache *nodeCache, unsigned int level) const;

  
  /**
     @brief 'bagCount' accessor.

     @return in-bag count for current tree.
   */
  inline unsigned int BagCount() const {
    return bagCount;
  }


  inline unsigned int SCount(int splitIdx) const {
    return indexNode[splitIdx].sCount;
  }


  inline void SetLH(unsigned int ptL) {
    ntLH[LevelOffPT(ptL)] = true;
  }
      

  inline void SetRH(unsigned int ptR) {
    ntRH[LevelOffPT(ptR)] = true;
  }


  inline unsigned int NextLH(int idxNext, unsigned int ptId, unsigned int _start, int idxCount, unsigned int sCount, double sum, double minInfo, unsigned char _path) {
    unsigned int pathNext = _path << 1;
    indexNode[idxNext].Init(idxNext, _start, ptId, idxCount, sCount, sum, minInfo, pathNext);

    SetLH(ptId);
    return pathNext;
  }

  
  inline unsigned int NextRH(int idxNext, int ptId, unsigned int _start, int idxCount, unsigned int sCount, double sum, double minInfo, unsigned char _path) {
    unsigned int pathNext = (_path << 1) | 1;
    indexNode[idxNext].Init(idxNext, _start, ptId, idxCount, sCount, sum, minInfo, pathNext);

    SetRH(ptId);
    return pathNext;
  }


  /**
     @brief Computes a level-relative offset for an indexed Pretree node.

     @param ptId is the node index, assumed to be at or above 'levelBase'.

     @return the level-relative offset.  A negative offset, in particular,
     distinguishes nodes belonging to earlier levels.
  */
  inline int LevelOffPT(unsigned int ptId) const {
    return ptId - levelBase;
  }


  /**
   @brief Returns the level-relative offset associated with an index node.

   @param splitIdx is the split index referenced.

   @return pretree offset from level base.
  */
  inline int LevelOffSplit(int splitIdx) const {
    return LevelOffPT(indexNode[splitIdx].ptId);
  }


  /**
     @return count of pretree nodes at current level.
  */
  inline unsigned int LevelWidth() const {
    return levelWidth;
  }

  bool LevelOffSample(unsigned int sIdx, unsigned int &levelOff) const;
};

#endif
