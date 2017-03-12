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
   predictors.  IndexSets of the index tree can be thought of as representing
   collections of sample indices. The two subnodes of a node, moreover, can
   be thought of as defining a bipartition of the parent's index collection.

   IndexSets only live within a single level.
*/
class IndexSet {
  double preBias; // Inf of information values eligible for splitting.
  unsigned int splitIdx; // Unique level identifier.
  //unsigned int parIdx; // Parent index, for use in path threading.
  unsigned int ptId; // Index of associated PTSerial node.
  unsigned int lhStart; // Start index of LH in buffer:  Swiss cheese.
  unsigned int extent; // # distinct indices in the node.
  unsigned int sCount;  // # samples subsumed by this node.
  double sum; // Sum of all responses in node.
  double minInfo; // Minimum acceptable information on which to split.
  unsigned char path; // Bitwise record of recent reaching L/R path.

  // Post-splitting fields:
  class SSNode *ssNode; // Nonzero iff split identified.
  unsigned int lhExtent; // Total indices over LH.
  unsigned int lhSCount; // Total samples cover LH.
  double lhSum; // Sum of responses over LH.
  
  double PrebiasReg();
  double PrebiasCtg(const double sumSquares[]);
  void Successor(std::vector<IndexSet> &indexNext, class Bottom *bottom, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) const;
  void SuccInit(Bottom *bottom, unsigned int _splitIdx, unsigned int _parIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path);

  
  inline unsigned int PathLeft() const {
    return path << 1;
  }


  inline unsigned int PathRight() const {
    return (path << 1) | 1;
  }


 public:
  static unsigned int minNode;
  IndexSet();

  void SplitCensus(SSNode *argMax, unsigned int &leafThis, unsigned int &lhSplitNext, unsigned int &rhSplitNext, unsigned int &idxExtent, unsigned int &idxLive);
  void Consume(class Bottom *bottom, class PreTree *preTree);
  void NonTerminal(class Bottom *bottom, class PreTree *preTree);
  void Terminal(class Bottom *bottom);
  void Replay(class Bottom *bottom, const class PreTree *preTree);
  void Produce(class Bottom *bottom, const class PreTree *preTree, std::vector<IndexSet> &indexNext) const;

  
  /**
    @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next level.
    
    MUST guarantee that no zero-length "splits" have been introduced.
    Not only are these nonsensical, but they are also dangerous, as they violate
    various assumptions about the integrity of the intermediate respresentation.

    @param _extent is the count of indices subsumed by the node.

    @return true iff the node subsumes more than minimal count of buffer elements.
  */
  static inline bool Splitable(unsigned int _extent) /*const*/ {
    return _extent >= minNode;
  }


  /**
     @return count of splitable nodes precipitated in next level:  0/1.
   */
  static inline unsigned int SplitAccum(unsigned int _extent, unsigned int &_idxLive) {
    if (Splitable(_extent)) {
      _idxLive += _extent;
      return 1;
    }
    else {
      return 0;
    }
  }

  
  /**
     @brief Outputs fields used by pre-bias computation.

     @param _sCount outputs the sample count.

     @param _sum outputs the sum

     @return void.
   */
  inline void PrebiasFields(unsigned int &_sCount, double &_sum) const {
    _sCount = sCount;
    _sum = sum;
  }

  
  /**
     @brief Accessor for 'preBias' field.

     @return reference to 'preBias' field.
  */
  void SetPrebias(double _preBias) {
    preBias = _preBias;
  }


  inline void PathCoords(unsigned int &_start, unsigned int &_extent) {
    _start = lhStart;
    _extent = extent;
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
  inline unsigned int Extent() const {
    return extent;
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

     @param _extent outputs the count of unique indices.

     @param _sCount outputs the total sample count.

     @param _sum outputs the sum.

     @return preBias, with output parameters.
  */
  double inline SplitFields(unsigned int &_lhStart, unsigned int &_extent, unsigned int &_sCount, double &_sum) const {
    _lhStart = lhStart;
    _extent = extent;
    _sCount = sCount;
    _sum = sum;
    return preBias;
  }


  /**
     @brief Sets fields with values used immediately following splitting.

     @param _idx is the index within the containing vector.

     @para _extent is the index count.

     @return void.
   */
  void Init(unsigned int _splitIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) {
    splitIdx = _splitIdx;
    sCount = _sCount;
    lhStart = _lhStart;
    extent = _extent;
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
};


/**
   @brief The index sets associated with nodes at a single subtree level.
 */
class IndexLevel {
  static unsigned int totLevels;
  std::vector<IndexSet> indexSet;
  const unsigned int bagCount;

  static class PreTree *OneTree(const class PMTrain *pmTrain, class Bottom *_bottom, int _nSamp, unsigned int _bagCount, double _bagSum);
  unsigned int SplitCensus(const std::vector<class SSNode *> &argMax, unsigned int &leafNext, unsigned int &idxExtent, unsigned int &idxLive);
  void Consume(class Bottom *bottom, class PreTree *preTree, unsigned int splitNext, unsigned int leafNext, unsigned int idxExtent, unsigned int idxLive, bool terminal);
  void Produce(class Bottom *bottom, class PreTree *preTree);


 public:
  static void Immutables(unsigned int _minNode, unsigned int _totLevels);
  static void DeImmutables();
  //  class PreTree *preTree;

  IndexLevel(int _nSamp, unsigned int _bagCount, double _sum);
  ~IndexLevel();

  static class PreTree **BlockTrees(const class PMTrain *pmTrain, class Sample **sampleBlock, int _treeBlock);
  void Levels(class Bottom *bottom, class PreTree *preTree);
  unsigned int STIdx(class Bottom *bottom, unsigned int splitIdx, unsigned int relIdx) const;
  
  /**
     @brief 'bagCount' accessor.

     @return in-bag count for current tree.
   */
  inline unsigned int BagCount() const {
    return bagCount;
  }


  inline unsigned int LevelCount() const {
    return indexSet.size();
  }


  inline unsigned int SCount(int splitIdx) const {
    return indexSet[splitIdx].SCount();
  }


  inline unsigned int Extent(unsigned int splitIdx) const {
    return indexSet[splitIdx].Extent();
  }
  

  inline unsigned int StartIdx(unsigned int splitIdx) const {
    return indexSet[splitIdx].Start();
  }


  inline double MinInfo(unsigned int splitIdx) const {
    return indexSet[splitIdx].MinInfo();
  }


  inline double SplitFields(unsigned int splitIdx, unsigned int &idxStart, unsigned int &extent, unsigned int &sCount, double &sum) const {
    return indexSet[splitIdx].SplitFields(idxStart, extent, sCount, sum);
  }


  inline void PrebiasFields(unsigned int splitIdx, unsigned int &sCount, double &sum) const {
    indexSet[splitIdx].PrebiasFields(sCount, sum);
  }


  inline void SetPrebias(unsigned int splitIdx, double preBias) {
    indexSet[splitIdx].SetPrebias(preBias);
  }
};

#endif
