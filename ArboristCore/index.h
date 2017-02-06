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
  unsigned int splitIdx; // Position within containing vector:  split index.
  unsigned int parIdx; // Parent index, for use in path threading.
  unsigned int ptId; // Index of associated PTSerial node.
  unsigned int lhStart; // Start index of LH in buffer:  Swiss cheese.
  unsigned int idxCount; // # distinct indices in the node.
  unsigned int sCount;  // # samples subsumed by this node.
  double sum; // Sum of all responses in node.
  double minInfo; // Minimum acceptable information on which to split.
  unsigned char path; // Bitwise record of recent reaching L/R path.

  // Post-splitting fields:
  class SSNode *ssNode; // Convenient to cache for LH/RH partition.
  unsigned int lhIdxCount; // Total indices over LH.
  unsigned int lhSCount; // Total samples cover LH.
  double lhSum; // Sum of responses over LH.
  unsigned int ptL; // LH index into pre-tree.
  unsigned int ptR; // RH index into pre-tree.

  double PrebiasReg();
  double PrebiasCtg(const double sumSquares[]);

  
  inline unsigned int PathLeft() const {
    return path << 1;
  }


  inline unsigned int PathRight() const {
    return (path << 1) | 1;
  }


 public:
  static unsigned int minNode;
  IndexNode();

  void SplitCensus(SSNode *argMax, unsigned int &leafThis, unsigned int &lhSplitNext, unsigned int &rhSplitNext, unsigned int &idxLive, unsigned int &idxMax);

  void NonTerminal(class PreTree *preTree, class SamplePred *samplePred, class Bottom *bottom);
  void Consume(class PreTree *preTree, class SamplePred *samplePred, class Bottom *bottom);
  void Produce(std::vector<IndexNode> &indexNext, unsigned int &posLeft, unsigned int &posRight) const ;

  /**
    @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next level.
    
    MUST guarantee that no zero-length "splits" have been introduced.
    Not only are these nonsensical, but they are also dangerous, as they violate
    various assumptions about the integrity of the intermediate respresentation.

    @param _idxCount is the count of indices subsumed by the node.

    @return true iff the node subsumes more than minimal count of buffer elements.
  */
  static inline bool Splitable(unsigned int _idxCount) /*const*/ {
    return _idxCount >= minNode;
  }


  static inline bool SplitAccum(unsigned int _idxCount, unsigned int &_idxLive, unsigned int &_idxMax) {
    if (Splitable(_idxCount)) {
      _idxMax = _idxCount > _idxMax ? _idxCount : _idxMax;
      _idxLive += _idxCount;
      return true;
    }
    else {
      return false;
    }
  }

  
  /**
     @brief Appends one hand of a split onto next level's node list, if splitable.

     @return void.
  */
  inline void SplitHand(std::vector<IndexNode> &indexNext, unsigned int &idx, unsigned int _sCount, unsigned int _idxCount, unsigned int _lhStart, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) const {
    if (Splitable(_idxCount)) {
      indexNext[idx].Init(idx, splitIdx, _sCount, _idxCount, _lhStart, _minInfo, _ptId, _sum, _path);
      idx++;
    }
  }

  
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


  inline void PathCoords(unsigned int &_start, unsigned int &_extent) {
    _start = lhStart;
    _extent = idxCount;
  }


  /**
     @return 'lhStart' field.
   */
  inline unsigned int Start() const {
    return lhStart;
  }

  
  /**
     @return index countl
   */
  inline unsigned int IdxCount() const {
    return idxCount;
  }


  inline unsigned int SCount() const {
    return sCount;
  }


  inline unsigned int PTId() const {
    return ptId;
  }
  

  /**
     @brief Exposes fields relevant for SplitPred methods.   N.B.:  Not all methods use all fields.

     @param _lhStart outputs the left-most index.

     @param _idxCount outputs the count of unique indices.

     @param _sCount outputs the total sample count.

     @param _sum outputs the sum.

     @return preBias, with output parameters.
  */
  double inline SplitFields(unsigned int &_lhStart, unsigned int &_idxCount, unsigned int &_sCount, double &_sum) const {
    _lhStart = lhStart;
    _idxCount = idxCount;
    _sCount = sCount;
    _sum = sum;
    return preBias;
  }


  /**
     @brief Sets fields with values used immediately following splitting.

     @param _idx is the index within the containing vector.

     @para _idxCount is the index count.

     @return void.
   */
  void Init(unsigned int _idx, unsigned int _parIdx, unsigned int _sCount, unsigned int _idxCount, unsigned int _lhStart, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) {
    splitIdx = _idx;
    parIdx = _parIdx;
    sCount = _sCount;
    idxCount = _idxCount;
    lhStart = _lhStart;
    minInfo = _minInfo;
    ptId = _ptId;
    sum = _sum;
    path = _path;
  }


  /**
     @brief Exposes minimum-information value for the node.

     @return minInfo value.
   */
  inline double MinInfo() const {
    return minInfo;
  }


  inline void PathFields(unsigned int &_parIdx, unsigned int &_path, unsigned int &_lhStart, unsigned int &_idxCount, unsigned int &_ptId) const {
    _parIdx = parIdx;
    _path = path;
    _lhStart = lhStart;
    _idxCount = idxCount;
    _ptId = ptId;
  }
};


class Index {
  static unsigned int totLevels;
  //  unsigned int lhSplitNext; // Count of LH nodes in next level.

  std::vector<IndexNode> indexNode;
  const unsigned int bagCount;
  unsigned int levelWidth; // Count of pretree nodes at frontier.
  static class PreTree *OneTree(const class PMTrain *pmTrain, class SamplePred *_samplePred, class Bottom *_bottom, int _nSamp, unsigned int _bagCount, double _bagSum);
  unsigned int LevelCensus(const std::vector<class SSNode*> &argMax, unsigned int &lhSplitNext, unsigned int &leafNext, unsigned int &idxLive, unsigned int &idxMax);
  void LevelConsume(unsigned int splitNext, unsigned int leafNext);
  void LevelProduce(class Bottom *bottom, unsigned int splitNext, unsigned int leafNext, unsigned int idxLive, unsigned int idxMax);


 public:
  static void Immutables(unsigned int _minNode, unsigned int _totLevels);
  static void DeImmutables();
  class SamplePred *samplePred;
  class PreTree *preTree;
  class Bottom *bottom;
  Index(class SamplePred *_samplePred, class PreTree *_preTree, class Bottom *_bottom, int _nSamp, unsigned int _bagCount, double _sum);
  ~Index();

  static class PreTree **BlockTrees(const class PMTrain *pmTrain, class Sample **sampleBlock, int _treeBlock);
  void SetPrebias();
  void Levels();
  
  /**
     @brief 'bagCount' accessor.

     @return in-bag count for current tree.
   */
  inline unsigned int BagCount() const {
    return bagCount;
  }


  inline unsigned int SCount(int splitIdx) const {
    return indexNode[splitIdx].SCount();
  }


  /**
     @return count of pretree nodes at current level.
  */
  inline unsigned int LevelWidth() const {
    return levelWidth;
  }

  bool LevelOffSample(unsigned int sIdx, unsigned int &levelOff) const;
  unsigned int LevelOffSplit(unsigned int splitIdx) const;
};

#endif
