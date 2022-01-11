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

  IndexT splitIdx; // Unique level identifier.
  IndexT ptId; // Index of associated pretree node.
  IndexRange bufRange;  // Positions within obs-part buffer:  Swiss cheese.

  IndexT sCount;  // # samples subsumed by this set.
  double sum; // Sum of all responses in set.
  double minInfo; // Split threshold:  reset after splitting.
  unsigned char path; // Bitwise record of recent reaching L/R path.
  vector<SumCount> ctgSum;  // Per-category response sums.

  // Post-splitting fields:  (Updated iff argMax nontrivial.)
  bool doesSplit; // Sticky.  Sets iff local conditions satisfied.
  bool unsplitable;  // Candidate found to have single response value.

  // Map position:  true successor if nonterminal or node if terminal.
  IndexT idxNext;
  
  // Revised per criterion, assumed registered in order.
  IndexT extentTrue; // Total indices over true branch.
  IndexT sCountTrue; // Total samples over true branch.
  double sumTrue; // Acummulates sum of true branch responses.

  // Whether node encoding is implicitly true:  defined iff doesSplit.
  // May be updated multiple times by successive criteria.  Final
  // criterion prevails, assuming criteria accrue conditionally.
  bool trueEncoding;

  // Precipitates setting of unsplitable in respective successor.
  bool trueExtinct;
  bool falseExtinct;
  
  // State repeatedly polled and/or updated by Reindex methods.  Hence
  // appropriate to cache.
  //
  IndexT ptTrue;
  IndexT ptFalse;

  PathT pathTrue;  // Fixed:  path to true successor, if any.
  PathT pathFalse; // Fixed:  " " false " ".

  vector<SumCount> ctgTrue; // Per-category sums inherited from criterion.

  
  /**
     @brief Initializes certain fields to a default terminal state.

     @param inatt is an inattainable value.
   */
  void initInattainable(IndexT inatt) {
    idxNext = inatt;
  }
  

  /**
     @brief Initializes index set as a successor node.
  */
  void succInit(class Frontier *frontier,
                const IndexSet* par,
                bool trueBranch);
  
public:

  IndexSet();


  static void immutables(IndexT minNode);

  
  static void deImmutables();

  /**
     @brief Initializes root set using sample summary.

     @param sample summarizes the tree's response sampling.
   */
  void initRoot(const class Sample* sample);


  /**
     @brief Updates branch state from criterion encoding.

     @param enc encapsulates the splitting criteria.
   */
  void update(const struct CritEncoding& enc);

  
  /**
     @brief Caches state necessary for reindexing and useful subsequently.
  */
  void nonterminal(const class SampleMap& smNext);

  
  /**
     @return Maximal informative split, if any.
   */
  void candMax(const vector<class SplitNux>& cand,
	       class SplitNux& argMaxNux) const;
  

  bool isInformative(const SplitNux* nux) const;
  

  /**
     @brief Sums each category for a node splitable in the upcoming level.

     @param[out] sumSquares outputs the response sum of squares, over categories.

     @return per-category sums for the node.
  */
  vector<double> sumsAndSquares(double& sumSquares);

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
     @brief Produces next level's LH and RH index sets for a split.

     @param indexNext is the crescent successor level of index sets.
  */
  void succHands(Frontier* frontier,
                 vector<IndexSet>& indexNext) const;


  void succHand(Frontier* frontier,
                vector<IndexSet>& indexNext,
                bool trueBranch) const;

  /**
     @param Determines pretree index of specified successor.

     @return pretree index determined.
   */
  IndexT getPTIdSucc(const class Frontier* frontier,
                     bool trueBranch) const;


  unsigned int getPath() const {
    return path;
  }

  
  /**
     @brief Determines terminality by checking split history.

     @return true iff the node did not split.
   */
  inline bool isTerminal() const {
    return !doesSplit;
  }
  
  
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


  /**
     @brief Successor indices precomputed from smNext.

     By convention, the false-branch successor is one index higher
     than that for the true branch.

     @param trueBranch is true iff true sense specified.

     @return successor index along specified branch sense.
   */
  inline auto getIdxSucc(bool trueBranch) const {
    return trueBranch ? idxNext : idxNext + 1;
  }


  inline auto getSumSucc(bool trueBranch) const {
    return trueBranch ? sumTrue : sum - sumTrue;
  }


  inline auto getPTSucc(bool trueBranch) const {
    return trueBranch ? ptTrue : ptFalse;
  }


  inline auto getPathSucc(bool trueBranch) const {
    return trueBranch ? pathTrue : pathFalse;
  }


  inline auto getSCountSucc(bool trueBranch) const {
    return trueBranch ? sCountTrue : sCount - sCountTrue;
  }

  inline auto getStartSucc(bool trueBranch) const {
    return trueBranch ? bufRange.getStart() : bufRange.getStart() + extentTrue;
  }


  inline auto getExtentSucc(bool trueBranch) const {
    return trueBranch ? extentTrue : bufRange.getExtent() - extentTrue;
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

  
  inline bool encodesTrue() const {
    return trueEncoding;
  }
};

#endif
