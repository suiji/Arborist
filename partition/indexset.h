// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file indexset.h

   @brief Contiguous subsets of the ObsPart buffer.

   @author Mark Seligman
 */

#ifndef PARTITION_INDEXSET_H
#define PARTITION_INDEXSET_H


#include "splitcoord.h"
#include "sumcount.h"
#include "replay.h"

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

  // Whether node is implicitly left:  defined iff doesSplit.
  // May be updated multiple times by successive criteria.  Final
  // criterion prevails, assuming criteria accrue conditionally.
  bool leftImpl;

  // State repeatedly polled and/or updated by Reindex methods.  Hence
  // appropriate to cache.
  //
  IndexT ptLeft;
  IndexT ptRight;
  IndexT succLeft; // Fixed:  level index of left successor, if any.
  IndexT succRight; // Fixed:  " " right " "
  IndexT offLeft; // Increases:  accumulating left offset.
  IndexT offRight; // " "                     right offset.
  unsigned char pathLeft;  // Fixed:  path to left successor, if any.
  unsigned char pathRight; // Fixed:  " " right " ".

  // These fields pertain only to non-splitting sets, so can be
  // overlaid with above via a union.
  IndexT succOnly; // Fixed:  successor IndexSet.
  IndexT offOnly; // Increases:  accumulating successor offset.

  vector<SumCount> ctgLeft; // Per-category sums inherited from criterion.

  
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

  
  void nontermReindex(const class Replay* replay,
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

     @param sumExpl is the explicit summand.

     @param ctgExpl are the explicit per-category sums and counts.

     @param leftExpl is true iff explicit hand is left.
   */

  inline void criterionLR(double sumExpl,
                          const vector<SumCount>& ctgExpl,
                          bool leftExpl) {
    sumL += (leftExpl ? sumExpl : sum - sumExpl);
    SumCount::incr(ctgLeft, leftExpl ? ctgExpl : SumCount::minus(ctgSum, ctgExpl));
    leftImpl = !leftExpl; // Final state is most recently registered.
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
  void reindex(const class Replay* replay,
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
     @brief Getter for split index.
   */
  inline auto getSplitIdx() const {
    return splitIdx;
  }

  
  /**
     @brief Getter for number of response categories.
   */
  inline auto getNCtg() const {
    return ctgSum.size();
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


  inline auto getBufRange() const {
    return bufRange;
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
  inline IndexT offspring(const class Replay* replay,
                          IndexT sIdx,
                          IndexT& pathSucc,
                          IndexT& ptSucc) {
    return doesSplit ? offspringLive(replay->senseLeft(sIdx, leftImpl), pathSucc, ptSucc) : offspringTerm(pathSucc, ptSucc);
  }

  
  /**
     @brief Set path and pretree successor of nonterminal.

     @param isLeft indicates branch sense.

     @return (possibly psuedo-) index of successor IndexSet.
   */
  inline IndexT offspringLive(bool isLeft,
                          IndexT& pathSucc,
                          IndexT& ptSucc) {
      pathSucc = getPathSucc(isLeft);
      ptSucc = getPTSucc(isLeft);
      return getIdxSucc(isLeft);
  }


  inline IndexT offspringTerm(IndexT& pathSucc,
			      IndexT& ptSucc) {
    pathSucc = 0; // Dummy:  overwritten by caller.
    ptSucc = ptId;
    return succOnly;
  }

  
  /**
     @brief As above, but also tracks (pseudo) successor indices.  State
     is side-effected, moreover, so must be invoked sequentially.
   */
  inline IndexT offspring(const class Replay* replay,
                          IndexT sIdx,
                          unsigned int& pathSucc,
                          IndexT& idxSucc,
                          IndexT& ptSucc) {
    if (doesSplit) {
      bool isLeft = replay->senseLeft(sIdx, leftImpl);
      idxSucc = getOffSucc(isLeft);
      return offspringLive(isLeft, pathSucc, ptSucc);
    }
    else {
      idxSucc = offOnly++;
      return offspringTerm(pathSucc, ptSucc);
    }
  }
};

#endif
