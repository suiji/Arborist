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

#include "typeparam.h"
#include <vector>


/**
   Index tree node fields associated with the response, viz., invariant across
   predictors.  IndexSets of the index tree can be thought of as representing
   collections of sample indices. The two subnodes of a node, moreover, can
   be thought of as defining a bipartition of the parent's index collection.

   IndexSets only live within a single level.
*/
class IndexSet {
  unsigned int splitIdx; // Unique level identifier.
  unsigned int ptId; // Index of associated PTSerial node.
  unsigned int lhStart; // Start position of LH in buffer:  Swiss cheese.
  unsigned int extent; // # distinct indices in the set.
  unsigned int sCount;  // # samples subsumed by this set.
  double sum; // Sum of all responses in set.
  double minInfo; // Split threshold:  reset after splitting.
  unsigned int relBase; // Local copy of indexLevel's value.
  unsigned char path; // Bitwise record of recent reaching L/R path.
  vector<class SumCount> ctgSum;  // Per-category response sums.

  // Post-splitting fields:  (Set iff argMax nontrivial.)
  bool doesSplit; // iff argMax nontrivial.
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
  vector<class SumCount> ctgExpl; // Per-category sums.
  bool leftExpl; // Fixed:  whether left split explicit (else right).

  // These fields pertain only to non-splitting sets, so can be
  // overlaid with above via a union.
  unsigned int succOnly; // Fixed:  successor iSet.
  unsigned int offOnly; // Increases:  accumulating successor offset.
  
  void successor(class IndexLevel *indexLevel,
                 vector<IndexSet> &indexNext,
                 class Bottom *bottom,
                 unsigned int _sCount,
                 unsigned int _lhStart,
                 unsigned int _extent,
                 double _minInfo,
                 unsigned int _ptId,
                 bool explHand) const;

  void succInit(IndexLevel *indexLevel,
                Bottom *bottom,
                unsigned int _splitIdx,
                unsigned int _parIdx,
                unsigned int _sCount,
                unsigned int _lhStart,
                unsigned int _extent,
                double _minInfo,
                unsigned int _ptId,
                double _sum,
                unsigned int _path,
                const vector<class SumCount> &_ctgSum,
                const vector<class SumCount> &_ctgExpl,
                bool explHand);

  void nontermReindex(const class BV *replayExpl,
                      class IndexLevel *index,
                      unsigned int idxLive,
                      vector<unsigned int> &succST);
  
 public:
  IndexSet();
  void Init(unsigned int _splitIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path, unsigned int _relBase, unsigned int bagCount, const vector<class SumCount> &_ctgTot, const vector<SumCount> &_ctgExpl, bool explHand);
  void decr(vector<class SumCount> &_ctgTot, const vector<class SumCount> &_ctgSub);

  void applySplit(const vector<class SplitCand> &argMax);
  
  void splitCensus(class IndexLevel *indexLevel, unsigned int &leafThis, unsigned int &splitNext, unsigned int &idxLive, unsigned int &idxMax);

  void consume(class IndexLevel *indexlevel,
               class Bottom *bottom,
               class PreTree *preTree,
               const vector<class SplitCand> &argMax);

  void nonTerminal(class IndexLevel *indexLevel,
                   class PreTree *preTree,
                   const class SplitCand &argMax);

  void terminal(class IndexLevel *indexLevel);

  void blockReplay(class SamplePred *samplePred,
                   const class SplitCand& argMax,
                   BV *replayExpl);

  void blockReplay(class SamplePred *samplePred,
                   const class SplitCand& argMax,
                   unsigned int blockStart,
                   unsigned int blockExtent,
                   BV *replayExpl);

  void reindex(const class BV *replayExpl,
               class IndexLevel *index,
               unsigned int idxLive,
               vector<unsigned int> &succST);
  void produce(class IndexLevel *indexLevel, class Bottom *bottom, const class PreTree *preTree, vector<IndexSet> &indexNext) const;
  static unsigned SplitAccum(class IndexLevel *indexLevel, unsigned int _extent, unsigned int &_idxLive, unsigned int &_idxMax);
  const vector<class SumCount> &CtgDiff();

  /**
     @brief Sums each category for a node splitable in the upcoming level.

     @param[out] sumSquares accumulates the sum of squares over each category.
     Assumed intialized to zero.

     @param[in, out] sumOut records the response sums, by category.  Assumed initialized to zero.

     @return void, with side-effected 'unsplitable' state.
  */
  void sumsAndSquares(double &sumSquares, double *sumOut);

  bool isUnsplitable() const {
    return unsplitable;
  }

  
  /**
   */
  inline unsigned int getSplitIdx() const {
    return splitIdx;
  }

  
  inline void PathCoords(unsigned int &start,
                         unsigned int &extent) {
    start = this->lhStart;
    extent = this->extent;
  }


  /**
     @return 'lhStart' field.
   */
  inline unsigned int getStart() const {
    return lhStart;
  }

  
  /**
     @brief Index node extent accessor.

     @return index extent.
   */
  inline unsigned int getExtent() const {
    return extent;
  }


  inline double getSum() const {
    return sum;
  }
  

  inline unsigned int getSCount() const {
    return sCount;
  }


  inline unsigned int getPTId() const {
    return ptId;
  }
  

  /**
     @brief Exposes fields relevant for SplitPred methods.   N.B.:  Not all methods use all fields.

     @param _lhStart outputs the left-most index.

     @param _extent outputs the count of unique indices.

     @param _sCount outputs the total sample count.

     @param _sum outputs the sum.

     @return void.
  */
  void inline getSplitFields(unsigned int &lhStart,
                            unsigned int &extent,
                            unsigned int &sCount,
                            double &sum) const {
    lhStart = this->lhStart;
    extent = this->extent;
    sCount = this->sCount;
    sum = this->sum;
  }


  /**
     @brief Exposes minimum-information value for the node.

     @return minInfo value.
   */
  inline double getMinInfo() const {
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
  inline unsigned int offspring(bool expl, unsigned int &pathSucc, unsigned int &ptSucc) {
    unsigned int iSetSucc;
    if (!doesSplit) {  // Terminal from previous level.
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
  inline unsigned int offspring(bool expl,
                                unsigned int &pathSucc,
                                unsigned int &idxSucc,
                                unsigned int &ptSucc) {
    idxSucc = !doesSplit ? offOnly++ : (expl ? offExpl++ : offImpl++);
    return offspring(expl, pathSucc, ptSucc);
  }
};


/**
   @brief The index sets associated with nodes at a single subtree level.
 */
class IndexLevel {
  static unsigned int minNode;
  static unsigned int totLevels;
  unique_ptr<class SamplePred> samplePred;
  vector<IndexSet> indexSet;
  const unsigned int bagCount;
  unique_ptr<class SplitNode> splitNode;
  unique_ptr<class Bottom> bottom;
  bool nodeRel; // Whether level uses node-relative indexing:  sticky.
  bool levelTerminal; // Whether this level must exit.
  unsigned int idxLive; // Total live indices.
  unsigned int liveBase; // Accumulates live index offset.
  unsigned int extinctBase; // Accumulates extinct index offset.
  unsigned int succLive; // Accumulates live indices for upcoming level.
  unsigned int succExtinct; // " " extinct "
  vector<unsigned int> relBase; // Node-to-relative index.
  vector<unsigned int> succBase; // Overlaps, then moves to relBase.
  vector<unsigned int> rel2ST; // Maps to subtree index.
  vector<unsigned int> rel2PT; // Maps to pretree index.
  vector<unsigned int> st2Split; // Useful for subtree-relative indexing.
  vector<unsigned int> st2PT; // Frontier map.
  class BV *replayExpl;

  unsigned int splitCensus(const vector<class SplitCand> &argMax,
                           unsigned int &leafNext,
                           unsigned int &idxMax,
                           bool _levelTerminal);

  void consume(class PreTree *preTree,
               const vector<class SplitCand> &argMax,
               unsigned int splitNext,
               unsigned int leafNext,
               unsigned int idxMax);

  void produce(const class PreTree *preTree,
               unsigned int splitNext);


 public:
  static void Immutables(unsigned int _minNode, unsigned int _totLevels);
  static void DeImmutables();

  IndexLevel(const class FrameTrain* frameTrain,
             const class RowRank* rowRank,
             const class Sample* sample);

  ~IndexLevel();

  /**
    @brief Performs sampling and level processing for a single tree.

    @param frameTrain contains the predictor frame mappings.

    @param sample contains the bagging summary.

    @param rowRank contains the per-predictor observation rankings.

    @return trained pretree object.
  */
  static shared_ptr<class PreTree> oneTree(const class FrameTrain* frameTrain,
                                           const class RowRank* rowRank,
                                           const class Sample* sample);


  /**
     @brief Main loop for per-level splitting.  Assumes root node and
     attendant per-tree data structures have been initialized.

     @param frameTrain holds the predictor cardinality values.

     @return trained pretree object.
  */
  shared_ptr<class PreTree> levels(const class FrameTrain *frameTrain,
                                   const class Sample* sample);
  
  bool nonTerminal(class PreTree *preTree,
                   IndexSet *iSet,
                   const class SplitCand &argMax);

  /**
     @brief Builds index base offsets to mirror crescent pretree level.

     @param extent is the count of the index range.

     @param ptId is the index of the corresponding pretree node.

     @param offOut outputs the node-relative starting index.  Should not
     exceed 'idxExtent', the live high watermark of the previous level.

     @param terminal is true iff predecessor node is terminal.

     @return successor index count.
  */
  unsigned int idxSucc(unsigned int extent,
                       unsigned int ptId,
                       unsigned int &outOff,
                       bool terminal = false);

  void blockReplay(IndexSet *iSet,
                   const class SplitCand& argMax,
                   unsigned int blockStart,
                   unsigned int blockExtent) const;

  /**
     @brief Dispatches nonterminal method based on predictor type.

     @param argMax is the split candidate characterizing the nonterminal.

     @param preTree is the crescent pretree.
  
     @param iSet is the node being split.

     @param run specifies the run sets associated with the node.

     @return true iff left-hand of split is explicit.
  */
  bool nonTerminal(const class SplitCand &argMax,
                   class PreTree *preTree,
                   class IndexSet *iSet,
                   class Run *run) const;

  bool replayRun(const class SplitCand &argMax,
                 class IndexSet *iSet,
                 class PreTree *preTree,
                 const class Run *run) const;

  bool branchNum(const class SplitCand& argMax,
                 class IndexSet *iSet,
                 class PreTree *preTree) const;

  /**
     @brief Drives node-relative re-indexing.
   */
  void nodeReindex();

  void subtreeReindex(unsigned int splitNext);

  void chunkReindex(class IdxPath *stPath,
                    unsigned int splitNext,
                    unsigned int chunkStart,
                    unsigned int chunkNext);
  void transitionReindex(unsigned int splitNext);

  unsigned int relLive(unsigned int relIdx,
                       unsigned int targIdx,
                       unsigned int path,
                       unsigned int base,
                       unsigned int ptIdx);
  void relExtinct(unsigned int relIdx, unsigned int ptId);

  void sumsAndSquares(unsigned int ctgWidth,
                      vector<double> &sumSquares,
                      vector<double> &ctgSum);


  /**
    @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next level.
    
    MUST guarantee that no zero-length "splits" have been introduced.
    Not only are these nonsensical, but they are also dangerous, as they violate
    various assumptions about the integrity of the intermediate respresentation.

    @param _extent is the count of indices subsumed by the node.

    @return true iff the node subsumes more than minimal count of buffer elements.
  */
  inline bool isSplitable(unsigned int extent) {
    return !levelTerminal && extent >= minNode;
  }


  /**
     @brief 'bagCount' accessor.

     @return in-bag count for current tree.
   */
  inline unsigned int getBagCount() const {
    return bagCount;
  }


  /**
     @brief Accessor for count of splitable nodes.
   */
  inline unsigned int getNSplit() const {
    return indexSet.size();
  }

  inline double getSum(unsigned int splitIdx) const {
    return indexSet[splitIdx].getSum();
  }

  
  inline unsigned int getSCount(unsigned int splitIdx) const {
    return indexSet[splitIdx].getSCount();
  }


  inline unsigned int getExtent(unsigned int splitIdx) const {
    return indexSet[splitIdx].getExtent();
  }
  

  inline unsigned int StartIdx(unsigned int splitIdx) const {
    return indexSet[splitIdx].getStart();
  }


  inline void getSplitFields(unsigned int splitIdx,
                            unsigned int &idxStart,
                            unsigned int &extent,
                            unsigned int &sCount,
                            double &sum) const {
    return indexSet[splitIdx].getSplitFields(idxStart, extent, sCount, sum);
  }


  inline unsigned int SuccBase(unsigned int splitIdx) {
    return succBase[splitIdx];
  }


  inline unsigned int RelBase(unsigned int splitIdx) {
    return relBase[splitIdx];
  }


  inline bool isUnsplitable(unsigned int splitIdx) const {
    return indexSet[splitIdx].isUnsplitable();
  }

  
/**
   @brief Dispatches consecutive node-relative indices to frontier map for
   final pre-tree node assignment.

   @return void.
 */
  void relExtinct(unsigned int relBase,
                  unsigned int extent,
                  unsigned int ptId) {
    for (unsigned int relIdx = relBase; relIdx < relBase + extent; relIdx++) {
      relExtinct(relIdx, ptId);
    }
  }


  /**
     @brief Reconciles remaining live node-relative indices.
  */
  void relFlush() {
    if (nodeRel) {
      for (unsigned int relIdx = 0; relIdx < idxLive; relIdx++) {
        relExtinct(relIdx, rel2PT[relIdx]);
      }
    }
  }
};

#endif
