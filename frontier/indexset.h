// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file indexset.h

   @brief Frontier nodes represented as contiguous subsets of the ObsPart buffer.

   @author Mark Seligman
 */

#ifndef FRONTIER_INDEXSET_H
#define FRONTIER_INDEXSET_H



#include "splitcoord.h"
#include "sumcount.h"
#include "branchsense.h"


/**
   Index tree node fields associated with the response, viz., invariant across
   predictors.  IndexSets of the index tree can be thought of as representing
   collections of sample indices. The two subnodes of a node, moreover, can
   be thought of as defining a bipartition of the parent's index collection.

   IndexSets only live within a single level.
*/
class IndexSet {
  static IndexT minNode;

  const IndexT splitIdx; // Unique level identifier.
  const IndexRange bufRange;  // Swiss cheese positions within obsPart buffer.
  const IndexT sCount;  // # samples subsumed by this set.
  const double sum; // Sum of all responses in set.
  const PathT path; // Bitwise record of recent reaching L/R path.
  const IndexT ptId; // Index of associated pretree node.
  const vector<SumCount> ctgSum;  // Per-category sum decomposition.

  double minInfo; // Split threshold:  reset after splitting.

  // Post-splitting fields:  (Updated iff argMax nontrivial.)
  bool doesSplit; // Sets iff local conditions satisfied.

  // Set by fiat or discovery.  E.g., candidate has single response value.
  bool unsplitable; 

  // Map position:  successor true index if nonterminal otherwise terminal index.
  IndexT idxNext;
  
  // Revised per criterion, assumed registered in order.
  IndexT extentTrue; // Total indices over true branch.
  IndexT sCountTrue; // Total samples over true branch.
  double sumTrue; // Acummulates sum of true branch responses.

  // Whether node encoding is implicitly true:  defined iff doesSplit.
  // May be updated multiple times by successive criteria.  Final
  // criterion prevails, assuming criteria accrue conditionally.
  bool trueEncoding;
  vector<SumCount> ctgTrue; // Per-category sums updatable from criterion.

  // Precipitates setting of unsplitable in respective successor.
  bool trueExtinct;
  bool falseExtinct;

public:

  /**
     @brief Root node constructor.
   */
  IndexSet(const class SampledObs* sample);


  /**
     @brief Successor node constructor.
   */
  IndexSet(const class Frontier *frontier,
	   const IndexSet& pred,
	   bool trueBranch);


  static void immutables(IndexT minNode);

  
  static void deImmutables();


  /**
     @brief Updates branch state from criterion encoding.

     @param enc encapsulates the splitting criteria.
   */
  void update(const struct CritEncoding& enc);

  
  /**
     @brief Selects best splitter, if any.

     @return maximal- or zero=information candidate for node.
   */
  class SplitNux candMax(const vector<class SplitNux>& cand) const;


  /**
     @return true iff minimum information threshold exceeded.
   */
  bool isInformative(const SplitNux& nux) const;
  

  /**
     @brief Sums each category for a node splitable in the upcoming level.

     @param[out] sumSquares outputs the response sum of squares, over categories.

     @return per-category sums for the node.
  */
  vector<double> sumsAndSquares(double& sumSquares);


  /**
     @brief Computes the successor path along the specified branch.
   */
  PathT getPathSucc(bool trueBranch) const;


  bool isUnsplitable() const {
    return unsplitable;
  }


  IndexT getIdxNext() const {
    return idxNext;
  }
  

  void setIdxNext(IndexT mapIdx) {
    idxNext = mapIdx;
  }
  
  
  /**
     @brief Sets state unsplitable.

     Used to terminate splitting loop gracefully.
   */
  void setUnsplitable() {
    unsplitable = true;
  }


  /**
     @brief Sets the respective successor extinction flag.

     @param senseTrue indicates the successor's branch sense.
   */
  void setExtinct(bool senseTrue) {
    if (senseTrue) {
      trueExtinct = true;
    }
    else {
      falseExtinct = true;
    }
  }


  void setExtinct() {
    setExtinct(true);
    setExtinct(false);
  }


  /**
     @brief Determines whether a given successor is scheduled for extinction.

     @param senseTrue indicates the successor's branch sense.
   */
  bool succExtinct(bool senseTrue) const {
    return senseTrue ? trueExtinct : falseExtinct;
  }


  /**
     @param Determines pretree index of specified successor.

     @return pretree index determined.
   */
  IndexT getPTIdSucc(const class Frontier* frontier,
                     bool trueBranch) const;


  PathT getPath(unsigned int mask) const {
    return path & mask;
  }

  
  /**
     @brief Determines terminality by checking split history.

     @return true iff the node did not split.
   */
  bool isTerminal() const {
    return !doesSplit;
  }
  
  
  /**
     @brief Getter for split index.
   */
  auto getSplitIdx() const {
    return splitIdx;
  }


  const vector<SumCount>& getCtgSumCount() const {
    return ctgSum;
  }


  const SumCount getCtgSumCount(CtgT ctg) const {
    return ctgSum[ctg];
  }


  const IndexT getCategoryCount(CtgT ctg) const {
    return ctgSum[ctg].sCount;
  }
  
  
  /**
     @brief Getter for number of response categories.
   */
  auto getNCtg() const {
    return ctgSum.size();
  }


  /**
     @brief Successor indices precomputed from smNext.

     By convention, the false-branch successor is one index higher
     than that for the true branch.

     @param trueBranch is true iff true sense specified.

     @return successor index along specified branch sense.
   */
  auto getIdxSucc(bool trueBranch) const {
    return trueBranch ? idxNext : idxNext + 1;
  }


  auto getSumSucc(bool trueBranch) const {
    return trueBranch ? sumTrue : sum - sumTrue;
  }


  auto getSCountSucc(bool trueBranch) const {
    return trueBranch ? sCountTrue : sCount - sCountTrue;
  }

  auto getStartSucc(bool trueBranch) const {
    return trueBranch ? bufRange.getStart() : bufRange.getStart() + extentTrue;
  }


  auto getExtentSucc(bool trueBranch) const {
    return trueBranch ? extentTrue : bufRange.getExtent() - extentTrue;
  }

  
  /**
     @brief Getters returning like-named member value.
   */

  auto getStart() const {
    return bufRange.getStart();
  }

  
  auto getExtent() const {
    return bufRange.getExtent();
  }


  auto getSum() const {
    return sum;
  }
  

  auto getSCount() const {
    return sCount;
  }


  auto getPTId() const {
    return ptId;
  }


  auto getBufRange() const {
    return bufRange;
  }
  
  
  /**
     @brief Exposes minimum-information value for the node.

     @return minInfo value.
  */
  auto getMinInfo() const {
    return minInfo;
  }

  
  bool encodesTrue() const {
    return trueEncoding;
  }
};

#endif
