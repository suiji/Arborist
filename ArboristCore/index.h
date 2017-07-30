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
  unsigned int lhStart; // Start position of LH in buffer:  Swiss cheese.
  unsigned int extent; // # distinct indices in the set.
  unsigned int sCount;  // # samples subsumed by this set.
  double sum; // Sum of all responses in set.
  double minInfo; // Split threshold:  reset after splitting.
  unsigned int relBase; // Local copy of indexLevel's value.
  unsigned char path; // Bitwise record of recent reaching L/R path.
  std::vector<class SumCount> ctgSum;  // Per-category response sums.

  // Post-splitting fields:  (Set iff argMax nontrivial.)
  bool terminal; // Whether argMax trivial.
  bool unsplitable;  // Candidate found to have single response value.
  unsigned int lhExtent; // Total indices over LH.
  unsigned int lhSCount; // Total samples over LH.
  double sumExpl; // Sum of explicit index responses.

  // State repeatedly polled and/or updated by Reindex methods.  Hence
  // appropriate to cache.
  //
  unsigned int ptExpl;
  unsigned int ptImpl;
  unsigned int succExpl; // Fixed:  level index of explicit successor, if any.
  unsigned int succImpl; // Fixed:  " " implicit " "
  unsigned int offExpl; // Increases:  accumulating explicit offset.
  unsigned int offImpl; // Increases:  accumulating implicit offset.
  unsigned char pathExpl;  // Fixed:  path to explicit successor, if any.
  unsigned char pathImpl; // Fixed:  path to implicit successor, if any.
  std::vector<class SumCount> ctgExpl; // Per-category sums.
  bool leftExpl; // Fixed:  whether left split explicit (else right).

  // These fields pertain only to non-splitting sets, so can be
  // overlaid with above via a union.
  unsigned int succOnly; // Fixed:  successor iSet.
  unsigned int offOnly; // Increases:  accumulating successor offset.
  
  void Successor(class IndexLevel *indexLevel, std::vector<IndexSet> &indexNext, class Bottom *bottom, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, bool explHand) const;
  void SuccInit(IndexLevel *indexLevel, Bottom *bottom, unsigned int _splitIdx, unsigned int _parIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path, const std::vector<class SumCount> &_ctgSum, const std::vector<class SumCount> &_ctgExpl, bool explHand);
  void NontermReindex(const class BV *replayExpl, class IndexLevel *index, unsigned int idxLive, std::vector<unsigned int> &succST);

  
 public:
  IndexSet();
  void Init(unsigned int _splitIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path, unsigned int _relBase, unsigned int bagCount, const std::vector<class SumCount> &_ctgTot, const std::vector<SumCount> &_ctgExpl, bool explHand);
  void Decr(std::vector<class SumCount> &_ctgTot, const std::vector<class SumCount> &_ctgSub);
  void ApplySplit(const std::vector<class SSNode> &argMax);
  void SplitCensus(class IndexLevel *indexLevel, unsigned int &leafThis, unsigned int &splitNext, unsigned int &idxLive, unsigned int &idxMax);
  void Consume(class IndexLevel *indexlevel, class Bottom *bottom, class PreTree *preTree, const std::vector<class SSNode> &argMax);
  void NonTerminal(class IndexLevel *indexLevel, class PreTree *preTree, const class SSNode &argMax);
  void Terminal(class IndexLevel *indexLevel);
  void BlockReplay(class SamplePred *samplePred, unsigned int predIdx, unsigned int bufIdx, unsigned int blockStart, unsigned int blockExtent, BV *replayExpl);
  void Reindex(const class BV *replayExpl, class IndexLevel *index, unsigned int idxLive, std::vector<unsigned int> &succST);
  void Produce(class IndexLevel *indexLevel, class Bottom *bottom, const class PreTree *preTree, std::vector<IndexSet> &indexNext) const;
  static unsigned SplitAccum(class IndexLevel *indexLevel, unsigned int _extent, unsigned int &_idxLive, unsigned int &_idxMax);
  const std::vector<class SumCount> &CtgDiff();
  void SetPrebias(const class Bottom *bottom);
  void SumsAndSquares(double &sumSquares, double *sumOut);

  bool Unsplitable() const {
    return unsplitable;
  }

  
  /**
   */
  inline unsigned int SplitIdx() const {
    return splitIdx;
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
     @brief Exposes minimum-information value for the node.

     @return minInfo value.
   */
  inline double MinInfo() const {
    return minInfo;
  }


  /**
     @brief L/R accessor for subtree-relative reindexing.

     @param expl is true iff the successor lies in the explicit side of
     the split.

     @param pathSucc outputs the (possibly pseudo) successor path.

     @param idxSucc outputs the (possibly pseudo) successor index.

     @return index (possibly pseudo) of successor index set.
   */
  inline unsigned int Offspring(bool expl, unsigned int &pathSucc, unsigned int &ptSucc) {
    unsigned int iSetSucc;
    if (terminal) {  // Terminal from previous level.
      iSetSucc = succOnly;
      ptSucc = ptId;
      pathSucc = 0; // Dummy:  overwritten by caller.
    }
    else {
      iSetSucc = expl ? succExpl : succImpl;
      pathSucc = expl ? pathExpl : pathImpl;
      ptSucc = expl ? ptExpl : ptImpl;
    }
    return iSetSucc;
  }

  
  /**
     @brief As above, but also tracks (pseudo) successor indices.  State
     is side-effected, moreover, so must be invoked sequentially.
   */
  inline unsigned int Offspring(bool expl, unsigned int &pathSucc, unsigned int &idxSucc, unsigned int &ptSucc) {
    idxSucc = terminal ? offOnly++ : (expl ? offExpl++ : offImpl++);
    return Offspring(expl, pathSucc, ptSucc);
  }
};


/**
   @brief The index sets associated with nodes at a single subtree level.
 */
class IndexLevel {
  static unsigned int minNode;
  static unsigned int totLevels;
  class SamplePred *samplePred;
  class Bottom *bottom;
  std::vector<IndexSet> indexSet;
  const unsigned int bagCount;
  bool nodeRel; // Whether level uses node-relative indexing:  sticky.
  bool levelTerminal; // Whether this level must exit.
  unsigned int idxLive; // Total live indices.
  unsigned int liveBase; // Accumulates live index offset.
  unsigned int extinctBase; // Accumulates extinct index offset.
  unsigned int succLive; // Accumulates live indices for upcoming level.
  unsigned int succExtinct; // " " extinct "
  std::vector<unsigned int> relBase; // Node-to-relative index.
  std::vector<unsigned int> succBase; // Overlaps, then moves to relBase.
  std::vector<unsigned int> rel2ST; // Maps to subtree index.
  std::vector<unsigned int> rel2PT; // Maps to pretree index.
  std::vector<unsigned int> st2Split; // Useful for subtree-relative indexing.
  std::vector<unsigned int> st2PT; // Frontier map.
  class BV *replayExpl;

  static class PreTree *OneTree(const class PMTrain *pmTrain, const class RowRank *rowRank, const class Sample *sample, const class Coproc *coproc);
  void InfoInit(std::vector<class SSNode> &argMax) const;
  unsigned int SplitCensus(const std::vector<class SSNode> &argMax, unsigned int &leafNext, unsigned int &idxMax, bool _levelTerminal);
  void Consume(class PreTree *preTree, const std::vector<class SSNode> &argMax, unsigned int splitNext, unsigned int leafNext, unsigned int idxMax);
  void Produce(class PreTree *preTree, unsigned int splitNext);


 public:
  static void Immutables(unsigned int _minNode, unsigned int _totLevels);
  static void DeImmutables();

  IndexLevel(class SamplePred *_samplePred, const std::vector<class SumCount> &ctgRoot, class Bottom *_bottom, unsigned int _nSamp, unsigned int _bagCount, double _bagSum);
  ~IndexLevel();

  static void TreeBlock(const class PMTrain *pmTrain, const RowRank *rowRank, const std::vector<class Sample*> &sampleBlock, const class Coproc *coproc, std::vector<class PreTree*> &ptBlock);
  void Levels(const class RowRank *rowRank, const class Sample *sample, class PreTree *preTree);
  bool NonTerminal(class PreTree *preTree, IndexSet *iSet, const class SSNode &argMax);
  unsigned int IdxSucc(unsigned int extent, unsigned int ptId, unsigned int &outOff, bool terminal = false);
  void BlockReplay(IndexSet *iSet, unsigned int predIdx, unsigned int bufIdx, unsigned int blockStart, unsigned int blockExtent);

  void NodeReindex();
  void SubtreeReindex(unsigned int splitNext);
  void ChunkReindex(class IdxPath *stPath, unsigned int splitNext, unsigned int chunkStart, unsigned int chunkNext);
  void TransitionReindex(unsigned int splitNext);

  unsigned int RelLive(unsigned int relIdx, unsigned int targIdx, unsigned int path, unsigned int base, unsigned int ptIdx);
  void RelExtinct(unsigned int relIdx, unsigned int ptId);


  void SetPrebias();
  void SumsAndSquares(unsigned int ctgWidth, std::vector<double> &sumSquares, std::vector<double> &ctgSum);


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
     @brief 'bagCount' accessor.

     @return in-bag count for current tree.
   */
  inline unsigned int BagCount() const {
    return bagCount;
  }


  inline unsigned int NSplit() const {
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


  inline double SplitFields(unsigned int splitIdx, unsigned int &idxStart, unsigned int &extent, unsigned int &sCount, double &sum) const {
    return indexSet[splitIdx].SplitFields(idxStart, extent, sCount, sum);
  }


  inline unsigned int SuccBase(unsigned int splitIdx) {
    return succBase[splitIdx];
  }


  inline unsigned int RelBase(unsigned int splitIdx) {
    return relBase[splitIdx];
  }


  inline bool Unsplitable(unsigned int splitIdx) const {
    return indexSet[splitIdx].Unsplitable();
  }

  
/**
   @brief Dispatches consecutive node-relative indices to frontier map for
   final pre-tree node assignment.

   @return void.
 */
  void RelExtinct(unsigned int relBase, unsigned int extent, unsigned int ptId) {
    for (unsigned int relIdx = relBase; relIdx < relBase + extent; relIdx++) {
      RelExtinct(relIdx, ptId);
    }
  }


  /**
     @brief Reconciles remaining live node-relative indices.
  */
  void RelFlush() {
    if (nodeRel) {
      for (unsigned int relIdx = 0; relIdx < idxLive; relIdx++) {
	RelExtinct(relIdx, rel2PT[relIdx]);
      }
    }
  }
};

#endif
