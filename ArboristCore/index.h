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
  unsigned int ptId; // Index of associated PTSerial node.
  unsigned int lhStart; // Start index of LH in buffer:  Swiss cheese.
  unsigned int extent; // # distinct indices in the node.
  unsigned int sCount;  // # samples subsumed by this node.
  double sum; // Sum of all responses in node.
  double minInfo; // Minimum acceptable information on which to split.
  unsigned int relBase; // Local copy of indexLevel's value.
  unsigned char path; // Bitwise record of recent reaching L/R path.

  // Post-splitting fields:  (Set iff ssNode nonzero.)
  class SSNode *ssNode; // Nonzero iff split identified.
  unsigned int lhExtent; // Total indices over LH.
  unsigned int lhSCount; // Total samples cover LH.
  double lhSum; // Sum of responses over LH.

  // State repeatedly polled and/or updated by Reindex methods.  Hence
  // appropriate to cache.
  //
  unsigned int succLeft; // Fixed:  level index of left successor, if any.
  unsigned int succRight; // Fixed:  " " right "
  unsigned int offExpl; // Increases:  accumulating explicit offset.
  unsigned int offImpl; // Increases:  accumulating implicit offset.
  unsigned char pathLeft;  // Fixed:  path to left successor, if any.
  unsigned char pathRight; // Fixed:  path to right successor, if any.
  bool leftExpl; // Fixed:  whether left split explicit (else right).

  // These fields pertain only to non-splitting sets, so can be
  // overlaid with above via a union.
  unsigned int succOnly; // Fixed:  successor iSet.
  unsigned int offOnly; // Increases:  accumulating successor offset.
  
  double PrebiasReg();
  double PrebiasCtg(const double sumSquares[]);
  void Successor(class IndexLevel *indexLevel, std::vector<IndexSet> &indexNext, unsigned int succIdx, class Bottom *bottom, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) const;
  void SuccInit(IndexLevel *indexLevel, Bottom *bottom, unsigned int _splitIdx, unsigned int _parIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path);
  void NonterminalReindex(class Bottom *bottom, class BV *replayExpl, unsigned int idxLive, const std::vector<unsigned int> &rel2ST, std::vector<unsigned int> &succST);
  void TerminalReindex(class Bottom *bottom, const std::vector<unsigned int> &rel2ST, std::vector<unsigned int> &succST);

  
 public:
  IndexSet();

  void SplitCensus(std::vector<class SSNode*> &argMax, class IndexLevel *indexLevel, unsigned int &leafThis, unsigned int &splitNext, unsigned int &idxLive, unsigned int &idxMax);
  void Consume(class IndexLevel *indexlevel, class Bottom *bottom, class PreTree *preTree);
  void NonTerminal(class IndexLevel *indexLevel, class Bottom *bottom, class PreTree *preTree);
  void Terminal(class IndexLevel *indexLevel, class Bottom *bottom);
  void Reindex(class Bottom *bottom, class BV *replayExpl, unsigned int idxLive, const std::vector<unsigned int> &rel2ST, std::vector<unsigned int> &succST);
  void Produce(class IndexLevel *indexLevel, class Bottom *bottom, const class PreTree *preTree, std::vector<IndexSet> &indexNext) const;
  static unsigned SplitAccum(class IndexLevel *indexLevel, unsigned int _extent, unsigned int &_idxLive, unsigned int &_idxMax);


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
  void Init(unsigned int _splitIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path, unsigned int _relBase, unsigned int bagCount) {
    splitIdx = _splitIdx;
    sCount = _sCount;
    lhStart = _lhStart;
    extent = _extent;
    minInfo = _minInfo;
    ptId = _ptId;
    sum = _sum;
    path = _path;
    relBase = _relBase;

    // Inattainable value.  Reset only when non-terminal:
    succLeft = succRight = offExpl = offImpl = bagCount;
  }


  /**
     @brief Exposes minimum-information value for the node.

     @return minInfo value.
   */
  inline double MinInfo() const {
    return minInfo;
  }


  /**
     @brief Sets successor values for nonterminal node.

     @param expl is true iff the successor lies in the explicit side of
     the split.

     @param pathSucc outputs the (possibly pseudo) successor path.

     @param idxSucc outputs the (possibly pseudo) successor index.

     @return index (possibly pseudo) of successor index set.
   */
  inline unsigned int Offspring(bool expl, unsigned int &pathSucc, unsigned int &idxSucc) {
    unsigned int iSetSucc;
    if (ssNode == 0) {  // Terminal from previous level.
      iSetSucc = succOnly;
      idxSucc = offOnly++;
      pathSucc = 0; // Dummy:  overwritten by caller.
    }
    else {
      bool isLeft= (expl && leftExpl) || !(expl || leftExpl);
      iSetSucc = isLeft ? succLeft : succRight;
      pathSucc = isLeft ? pathLeft : pathRight;
      idxSucc = expl ? offExpl++ : offImpl++;
    }
    return iSetSucc;
  }
};


/**
   @brief The index sets associated with nodes at a single subtree level.
 */
class IndexLevel {
  static unsigned int minNode;
  static unsigned int totLevels;
  std::vector<IndexSet> indexSet;
  const unsigned int bagCount;
  bool levelTerminal; // Whether this level must exit.
  unsigned int idxLive; // Total live indices.
  unsigned int idxMax; // Widest live node.
  unsigned int liveBase; // Accumulates live index offset.
  unsigned int extinctBase; // Accumulates extinct index offset.
  unsigned int succLive; // Accumulates live indices for upcoming level.
  unsigned int succExtinct; // " " extinct "
  std::vector<unsigned int> relBase; // Node-to-relative index.
  std::vector<unsigned int> succBase; // Overlaps, then moves to relBase.
  std::vector<unsigned int> rel2ST; // Maps to subtree index.
  std::vector<unsigned int> succST;  // Overlaps, moves to, rel2ST.
  std::vector<unsigned int> st2Split; // Useful for subtree-relative indexing.

  static class PreTree *OneTree(const class PMTrain *pmTrain, class Bottom *_bottom, int _nSamp, unsigned int _bagCount, double _bagSum);
  unsigned int SplitCensus(std::vector<class SSNode *> &argMax, unsigned int &leafNext, bool _levelTerminal);
  void Consume(class Bottom *bottom, class PreTree *preTree, unsigned int splitNext, unsigned int leafNext);
  void Produce(class Bottom *bottom, class PreTree *preTree, unsigned int splitNext);


 public:
  static void Immutables(unsigned int _minNode, unsigned int _totLevels);
  static void DeImmutables();

  IndexLevel(int _nSamp, unsigned int _bagCount, double _sum);
  ~IndexLevel();

  static class PreTree **BlockTrees(const class PMTrain *pmTrain, class Sample **sampleBlock, int _treeBlock);
  void Levels(class Bottom *bottom, class PreTree *preTree);
  unsigned int IdxSucc(class Bottom *bottom, unsigned int extent, unsigned int ptId, unsigned int &outOff, bool terminal = false);
  void Reindex(class Bottom *bottom, class BV *replayExpl);
  void Reindex(class Bottom *bottom, class BV *replayExpl, class IdxPath *stPath);


  /**
   */
  unsigned int BagCount() {
    return bagCount;
  }
  
  
  /**
    @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next level.
    
    MUST guarantee that no zero-length "splits" have been introduced.
    Not only are these nonsensical, but they are also dangerous, as they violate
    various assumptions about the integrity of the intermediate respresentation.

    @param _extent is the count of indices subsumed by the node.

    @return true iff the node subsumes more than minimal count of buffer elements.
  */
  inline bool Splitable(unsigned int extent) {
    return !levelTerminal && extent >= minNode;
  }


/**
   @brief Looks up subtree-relative index from node-relative coordinates.

   @param splitIdx is an IndexSet index.

   @param relIdx is an offset relative to the set index.

   @return subtree-relative index.
 */
  inline unsigned int STIdx(unsigned int splitIdx, unsigned int relIdx) const {
    return rel2ST[relBase[splitIdx] + relIdx];
  }

  
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


  inline unsigned int SuccBase(unsigned int splitIdx) {
    return succBase[splitIdx];
  }


  inline unsigned int RelBase(unsigned int splitIdx) {
    return relBase[splitIdx];
  }
};

#endif
