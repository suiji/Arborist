// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file frontier.h

   @brief Partitions tree frontier, typically breadth-first.

   @author Mark Seligman

 */

#ifndef PARTITION_FRONTIER_H
#define PARTITION_FRONTIER_H

#include "splitcoord.h"
#include "sumcount.h"
#include "typeparam.h"
#include "bv.h"

#include <vector>

/**
   Index tree node fields associated with the response, viz., invariant across
   predictors.  IndexSets of the index tree can be thought of as representing
   collections of sample indices. The two subnodes of a node, moreover, can
   be thought of as defining a bipartition of the parent's index collection.

   IndexSets only live within a single level.
*/
class IndexSet {
  IndexT splitIdx; // Unique level identifier.
  IndexT ptId; // Index of associated pretree node.
  IndexRange bufRange;  // Positions within obs-part buffer:  Swiss cheese.

  IndexT sCount;  // # samples subsumed by this set.
  double sum; // Sum of all responses in set.
  double minInfo; // Split threshold:  reset after splitting.
  IndexT relBase; // Local copy of frontier's value.
  unsigned char path; // Bitwise record of recent reaching L/R path.
  vector<SumCount> ctgSum;  // Per-category response sums.

  // Post-splitting fields:  (Updated iff argMax nontrivial.)
  bool doesSplit; // iff local conditions satisfied.
  bool unsplitable;  // Candidate found to have single response value.
  IndexT lhExtent; // Total indices over LH.
  IndexT lhSCount; // Total samples over LH.

  // Revised per criterion, assumed registered in order.
  double sumL; // Acummulates sum of left index responses.
  bool leftImpl;  // Revised many times.  Last set value wins.

  // State repeatedly polled and/or updated by Reindex methods.  Hence
  // appropriate to cache.
  //
  IndexT ptLeft;
  IndexT ptRight;
  IndexT succLeft; // Fixed:  level index of explicit successor, if any.
  IndexT succRight; // Fixed:  " " implicit " "
  IndexT offLeft; // Increases:  accumulating explicit offset.
  IndexT offRight; // Increases:  accumulating implicit offset.
  unsigned char pathLeft;  // Fixed:  path to explicit successor, if any.
  unsigned char pathRight; // Fixed:  path to implicit successor, if any.

  // These fields pertain only to non-splitting sets, so can be
  // overlaid with above via a union.
  unsigned int succOnly; // Fixed:  successor iSet.
  unsigned int offOnly; // Increases:  accumulating successor offset.

  vector<SumCount> ctgLeft; // Per-category sums.

  
  /**
     @brief Initializes certain fields to a default terminal state.

     @param inatt is an inattainable value.
   */
  void initInattainable(IndexT inatt) {
    succLeft = succRight = offLeft = offRight = inatt;
  }
  
  /**
     @brief Initializes index set as a successor node.
  */
  void succInit(class Frontier *frontier,
                const IndexSet* par,
                bool isLeft);

  
  void nontermReindex(const class BV* replayExpl,
                      const class BV* replayLeft,
                      class Frontier *index,
                      IndexT idxLive,
                      vector<IndexT> &succST);
  
  /**
     @brief Caches state necessary for reindexing and useful subsequently.
  */
  void nonterminal(class Frontier *frontier);


  /**
     @brief Dispatches index set to frontier.

     @param frontier holds the partitioned data.
  */
  void terminal(class Frontier *frontier);


 public:
  IndexSet();


  /**
     @brief Initializes root set using sample summary.

     @param sample summarizes the tree's response sampling.
   */
  void initRoot(const class Sample* sample);


  /**
     @brief Revises L/R state according to criterion characteristics.

     @param sumExpl is an explicit summand.

     @param leftExpl is true iff explicit hand is left.
   */
  inline void criterionLR(double sumExpl,
                          vector<SumCount>& ctgExpl,
                          bool leftExpl) {
    leftImpl = !leftExpl; // Final state is most recently registered.
    sumL += leftExpl ? sumExpl : sum - sumExpl;
    SumCount::incr(ctgLeft, leftExpl ? ctgExpl : SumCount::minus(ctgSum, ctgExpl));
  }
  

  /**
     @brief Updates splitting state supplied by a criterion.
   */
  inline void consumeCriterion(double minInfo,
                               IndexT lhSCount,
                               IndexT lhExtent) {
    this->doesSplit = true;
    this->minInfo = minInfo;
    this->lhSCount += lhSCount;
    this->lhExtent += lhExtent;
  }


  /**
     @brief Dispatches according to terminal/nonterminal state.
   */
  void dispatch(class Frontier* frontier);
  
  /**
     @brief Node-relative reindexing:  indices contiguous on nodes (index sets).
  */
  void reindex(const class BV* replayExpl,
               const class BV* replayLeft,
               class Frontier *index,
               IndexT idxLive,
               vector<IndexT> &succST);


  /**
     @brief Sums each category for a node splitable in the upcoming level.

     @param[out] sumSquares outputs the response sum of squares, over categories.

     @return per-category sums for the node.
  */
  vector<double> sumsAndSquares(double& sumSquares);

  bool isUnsplitable() const {
    return unsplitable;
  }


  /**
     @brief Produces next level's LH and RH index sets for a split.

     @param indexNext is the crescent successor level of index sets.
  */
  void succHands(Frontier* frontier,
                 vector<IndexSet>& indexNext) const;


  void succHand(Frontier* frontier,
                vector<IndexSet>& indexNext,
                bool isLeft) const;

  /**
     @param Determines pretree index of specified successor.

     @return pretree index determined.
   */
  IndexT getPTIdSucc(const class Frontier* frontier,
                     bool isLeft) const;

  
  
  /**
     @param replayExpl bit set iff sample is explictly replayed.

     @param leftExpl defined iff sample also replayed:  L/R as defined.

     @param sIdx indexes the sample in question.

     @return true iff sample index is assigned to the left successor.
   */
  inline bool senseLeft(const class BV* replayExpl,
                        const class BV* replayLeft,
                        IndexT sIdx) const {
    return replayExpl->testBit(sIdx) ? replayLeft->testBit(sIdx) : leftImpl;
  }


  /**
     @brief Getter for split index.
   */
  inline auto getSplitIdx() const {
    return splitIdx;
  }


  inline const vector<SumCount>& getCtgSum() const {
    return ctgSum;
  }


  inline const vector<SumCount>& getCtgLeft() const {
    return ctgLeft;
  }


  inline auto getIdxSucc(bool isLeft) const {
    return isLeft ? succLeft : succRight;
  }


  inline auto getSumSucc(bool isLeft) const {
    return isLeft ? sumL : sum - sumL;
  }


  /**
     N.B.:  offset side effected.
   */
  inline auto getOffSucc(bool isLeft) {
    return isLeft ? offLeft++ : offRight++;
  }


  inline auto getPTSucc(bool isLeft) const {
    return isLeft ? ptLeft : ptRight;
  }


  inline auto getPathSucc(bool isLeft) const {
    return isLeft ? pathLeft : pathRight;
  }


  inline auto getSCountSucc(bool isLeft) const {
    return isLeft ? lhSCount : sCount - lhSCount;
  }

  inline auto getStartSucc(bool isLeft) const {
    return isLeft ? bufRange.getStart() : bufRange.getStart() + lhExtent;
  }


  inline auto getExtentSucc(bool isLeft) const {
    return isLeft ? lhExtent : bufRange.getExtent() - lhExtent;
  }

  
  /**
     @brief Getters returning like-named member value.
   */

  inline auto getStart() const {
    return bufRange.getStart();
  }

  
  inline auto getExtent() const {
    return bufRange.getExtent();
  }


  inline auto getSum() const {
    return sum;
  }
  

  inline auto getSCount() const {
    return sCount;
  }


  inline auto getPTId() const {
    return ptId;
  }

  
  /**
     @brief Exposes minimum-information value for the node.

     @return minInfo value.
   */
  inline auto getMinInfo() const {
    return minInfo;
  }

  

  /**
     @brief L/R accessor for subtree-relative reindexing.

     @param isExpl is true iff sample index tagged explicit.

     @param explLeft is true iff index both tagged explicitly left.

     @param pathSucc outputs the (possibly pseudo) successor path.

     @param idxSucc outputs the (possibly pseudo) successor index.

     @return index (possibly pseudo) of successor index set.
   */
  inline IndexT offspring(const BV* replayExpl,
                          const BV* replayLeft,
                          IndexT sIdx,
                          IndexT& pathSucc,
                          IndexT& ptSucc) {
    return offspring(senseLeft(replayExpl, replayLeft, sIdx), pathSucc, ptSucc);
  }

  inline IndexT offspring(bool isLeft,
                          IndexT& pathSucc,
                          IndexT& ptSucc) {
    if (!doesSplit) {  // Terminal from previous level.
      pathSucc = 0; // Dummy:  overwritten by caller.
      ptSucc = ptId;
      return succOnly;
    }
    else {
      pathSucc = getPathSucc(isLeft);
      ptSucc = getPTSucc(isLeft);
      return getIdxSucc(isLeft);
    }
  }

  
  /**
     @brief As above, but also tracks (pseudo) successor indices.  State
     is side-effected, moreover, so must be invoked sequentially.
   */
  inline IndexT offspring(const BV* replayExpl,
                          const BV* replayLeft,
                          IndexT sIdx,
                          unsigned int& pathSucc,
                          IndexT& idxSucc,
                          IndexT& ptSucc) {
    bool isLeft = senseLeft(replayExpl, replayLeft, sIdx);
    idxSucc = !doesSplit ? offOnly++ : getOffSucc(isLeft);
    return offspring(isLeft, pathSucc, ptSucc);
  }
};


/**
   @brief The index sets associated with nodes at a single subtree level.
 */
class Frontier {
  static unsigned int minNode;
  static unsigned int totLevels;
  vector<IndexSet> indexSet;
  const IndexT bagCount;
  unique_ptr<class Bottom> bottom;
  bool nodeRel; // Whether level uses node-relative indexing:  sticky.
  bool levelTerminal; // Whether this level must exit.
  IndexT idxLive; // Total live indices.
  IndexT liveBase; // Accumulates live index offset.
  IndexT extinctBase; // Accumulates extinct index offset.
  IndexT succLive; // Accumulates live indices for upcoming level.
  IndexT succExtinct; // " " extinct "
  vector<IndexT> relBase; // Node-to-relative index.
  vector<unsigned int> succBase; // Overlaps, then moves to relBase.
  vector<unsigned int> rel2ST; // Node-relative mapping to subtree index.
  vector<unsigned int> rel2PT; // Node-relative mapping to pretree index.
  vector<unsigned int> st2Split; // Subtree-relative mapping to split index.
  vector<unsigned int> st2PT; // Subtree-relative mapping to pretree index.
  unique_ptr<class BV> replayExpl; // Whether index is explicity replayed.
  unique_ptr<class BV> replayLeft; // If explicit, whether L/R; else undefined.
  unique_ptr<class PreTree> pretree; // Augmented per frontier.
  
  
  /**
     @brief Applies splitting results to new level.

     @param argMax are the per-node splitting candidates.

     @param levelTerminal_ indicates whether new level marked as final.
  */
  vector<IndexSet> splitDispatch(class SplitFrontier* splitFrontier,
                                 unsigned int level);

  /**
     @brief Establishes splitting parameters for next frontier level.
   */
  void nextLevel(const struct SplitSurvey& survey);
  

  /**
     @brief Resets parameters for upcoming levl.

     @param splitNext is the number of splits in the new level.
  */
  void reset(IndexT splitNext);


  /**
     @brief Reindexes by level modes: node-relative, subtree-relative, mixed.

     Parameters as above.
   */
  void reindex(const struct SplitSurvey& survey);

  
  /**
     @brief Produces new level's index sets and dispatches extinct nodes to pretree frontier.

     Parameters as described above.

     @return next level's splitable index set.
  */
  vector<IndexSet> produce(IndexT splitNext);


 public:

  /**
     @brief Initializes static invariants.

     @param minNode_ is the minimum node size for splitting.

     @param totLevels_ is the maximum number of levels to evaluate.
  */
  static void immutables(unsigned int minNode_,
                         unsigned int totLevels_);


  /**
     @brief Resets statics to default values.
  */
  static void deImmutables();


  /**
     @brief Per-tree constructor.  Sets up root node for level zero.
  */
  Frontier(const class SummaryFrame* frame,
           const class Sample* sample);

  ~Frontier();

  /**
    @brief Trains one tree.

    @param summaryFrame contains the predictor type mappings.

    @param sample contains the bagging summary.

    @return trained pretree object.
  */
  static unique_ptr<class PreTree> oneTree(const class SummaryFrame* frame,
                                           const class Sample* sample);


  /**
     @brief Drives breadth-first splitting.

     Assumes root node and attendant per-tree data structures have been initialized.
     Parameters as described above.
  */
  unique_ptr<class PreTree> levels(const class Sample* sample,
                                   class SplitFrontier* splitFrontier);
  

  /**
     @brief Counts offspring of this node, assumed not to be a leaf.

     @return count of offspring nodes.
  */
  unsigned int splitCensus(const IndexSet& iSet,
                           struct SplitSurvey& survey);


  /**
     @brief Accumulates index parameters of successor level.

     @param succExent is the index of extent of the putative successor set.

     @param[out] idxMax outputs the maximum successor index.

     @return count of splitable sets precipitated in next level:  0 or 1.
  */
  unsigned int splitAccum(IndexT succExtent,
                          struct SplitSurvey& survey);



  /**
     @brief Builds index base offsets to mirror crescent pretree level.

     @param extent is the count of the index range.

     @param ptId is the index of the corresponding pretree node.

     @param offOut outputs the node-relative starting index.  Should not
     exceed 'idxExtent', the live high watermark of the previous level.

     @param terminal is true iff predecessor node is terminal.

     @return successor index count.
  */
  IndexT idxSucc(IndexT extent,
                       IndexT ptId,
                       IndexT &outOff,
                       bool terminal = false);


  /**
     @brief PreTree pass-through to obtain successor index.

     @param ptId is the parent pretree index.

     @param isLeft indicates the successor handedness.

     @return successor index.
   */
  IndexT getPTIdSucc(IndexT ptId,
                     bool isLeft) const;
  

  /**
     @brief Bottom pass-through to register reaching path.

     @param splitIdx is the level-relative node index.

     @param parIdx is the parent node's index.

     @param bufRange is the subsumed buffer range.

     @param relBase is the index base.

     @param path is the inherited path.
   */
  void reachingPath(IndexT splitIdx,
                    IndexT parIdx,
                    const IndexRange& bufRange,
                    IndexT relBase,
                    unsigned int path) const;
  
  /**
     @brief Drives node-relative re-indexing.
   */
  void nodeReindex();

  /**
     @brief Subtree-relative reindexing:  indices randomly distributed
     among nodes (i.e., index sets).
  */
  void stReindex(IndexT splitNext);

  /**
     @brief Updates the split/path/pretree state of an extant index based on
     its position in the next level (i.e., left/right/extinct).

     @param stPath is a subtree-relative path.
  */
  void stReindex(class IdxPath *stPath,
                 IndexT splitNext,
                 IndexT chunkStart,
                 IndexT chunkNext);

  /**
     @brief As above, but initializes node-relative mappings for subsequent
     levels.  Employs accumulated state and cannot be parallelized.
  */
  void transitionReindex(IndexT splitNext);

  /**
     @brief Updates the mapping from live relative indices to associated
     PreTree indices.

     @return corresponding subtree-relative index.
  */
  IndexT relLive(IndexT relIdx,
                       IndexT targIdx,
                       IndexT path,
                       IndexT base,
                       IndexT ptIdx);
  /**
     @brief Translates node-relative back to subtree-relative indices on 
     terminatinal node.

     @param relIdx is the node-relative index.

     @param ptId is the pre-tree index of the associated node.
  */
  void relExtinct(IndexT relIdx,
                  IndexT ptId);

  
  /**
     @brief Visits all live indices, so likely worth parallelizing.

     @param[out] ctgSum outputs the per-category sum of responses, per node.

     @return category-summed squares of responses, per node.
  */
  vector<double> sumsAndSquares(vector<vector<double> >&ctgSum);


  /**
    @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next level.
    
    MUST guarantee that no zero-length "splits" have been introduced.
    Not only are these nonsensical, but they are also dangerous, as they violate
    various assumptions about the integrity of the intermediate respresentation.

    @param extent is the count of indices subsumed by the node.

    @return true iff the node subsumes more than minimal count of buffer elements.
  */
  inline bool isSplitable(IndexT extent) const {
    return !levelTerminal && extent >= minNode;
  }


  /**
     @brief Getter for # of distinct in-bag samples.

     @return bagCount value.
   */
  inline auto getBagCount() const {
    return bagCount;
  }


  /**
     @brief Getter for IndexSet at a given coordinate.

     @param splitCoord is the coordinate.

     @return reference to set at coordinate.
   */
  inline const IndexSet& getISet(const SplitCoord& splitCoord) const {
    return indexSet[splitCoord.nodeIdx];
  }
  
  /**
     @brief Accessor for count of splitable sets.
   */
  inline IndexT getNSplit() const {
    return indexSet.size();
  }

  /**
     @brief Accessor for sum of sampled responses over set.

     @param splitIdx is the level-relative index of a set.

     @return index set's sum value.
   */
  inline auto getSum(IndexT splitIdx) const {
    return indexSet[splitIdx].getSum();
  }


  /**
     @brief Accessor for count of sampled responses over set.
   */
  inline auto getSCount(IndexT splitIdx) const {
    return indexSet[splitIdx].getSCount();
  }


  /**
     @brief Accessor for count of disinct indices over set.
   */
  inline auto getExtent(IndexT splitIdx) const {
    return indexSet[splitIdx].getExtent();
  }
  

  /**
     @brief Accessor for relative base of split.
   */
  inline auto getRelBase(IndexT splitIdx) const {
    return relBase[splitIdx];
  }


  /**
     @brief Indicates whether index set is inherently unsplitable.
   */
  inline bool isUnsplitable(IndexT splitIdx) const {
    return indexSet[splitIdx].isUnsplitable();
  }

  
  /**
     @brief Dispatches consecutive node-relative indices to frontier map for
     final pre-tree node assignment.
  */
  void relExtinct(IndexT relBase,
                  IndexT extent,
                  IndexT ptId) {
    for (IndexT relIdx = relBase; relIdx < relBase + extent; relIdx++) {
      relExtinct(relIdx, ptId);
    }
  }


  /**
     @brief Reconciles remaining live node-relative indices.
  */
  void relFlush() {
    if (nodeRel) {
      for (IndexT relIdx = 0; relIdx < idxLive; relIdx++) {
        relExtinct(relIdx, rel2PT[relIdx]);
      }
    }
  }
};

#endif
