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
  IndexT relBase; // Local copy of frontier's value.
  unsigned char path; // Bitwise record of recent reaching L/R path.
  vector<SumCount> ctgSum;  // Per-category response sums.

  // Post-splitting fields:  (Updated iff argMax nontrivial.)
  bool doesSplit; // Sticky.  Sets iff local conditions satisfied.
  bool unsplitable;  // Candidate found to have single response value.

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
  IndexT succTrue; // Fixed:  level index of true successor, if any.
  IndexT succFalse; // Fixed:  " " false " "
  IndexT offTrue; // Increases:  accumulating true offset.
  IndexT offFalse; // " "                     false offset.
  PathT pathTrue;  // Fixed:  path to true successor, if any.
  PathT pathFalse; // Fixed:  " " false " ".

  // These fields pertain only to non-splitting sets, so can be
  // overlaid with above via a union.
  IndexT succOnly; // Fixed:  successor IndexSet.
  IndexT offOnly; // Increases:  accumulating successor offset.

  vector<SumCount> ctgTrue; // Per-category sums inherited from criterion.

  
  /**
     @brief Initializes certain fields to a default terminal state.

     @param inatt is an inattainable value.
   */
  void initInattainable(IndexT inatt) {
    succTrue = succFalse = offTrue = offFalse = inatt;
  }
  
  /**
     @brief Initializes index set as a successor node.
  */
  void succInit(class Frontier *frontier,
                const IndexSet* par,
                bool trueBranch);

  
  void nontermReindex(const class BranchSense* branchSense,
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

  
  /**
     @brief Counts offspring of this node, assumed not to be a leaf.

     @param[out] survey accumaltes the census values.

     @return count of offspring nodes.
  */
  unsigned int splitCensus(struct SplitSurvey& survey) const;
  

  /**
     @brief Accumulates index parameters of successor level.

     @param sense is the branch sense of the successor.

     @param[out] survey accumulates the census.

     @return count of splitable sets precipitated in next level:  0 or 1.
  */
  unsigned int splitAccum(bool sense,
			  struct SplitSurvey& survey) const;
  

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
     @brief Revises true branch state from nux's true state.

     @param nux encapsulates the splitting description.

     @param ctgExpl are the explicit per-category sums and counts.
   */
  void true2True(const class SplitNux* nux,
		 vector<SumCount>& ctgExpl);
  

  /**
     @brief As above, but revises true branch state from nux's encoded state.
   */
  void encoded2True(const class SplitNux* nux,
		    vector<SumCount>& ctgExpl);


  /**
     @brief Accumulates leaf and split counts.

     @param levelTerminal indicates whether the current level is terminal.

     @param[in, out] survey accumulates the census.
   */
  void surveySplit(bool levelTerminal,
		   struct SplitSurvey& survey) const;
  
  bool isInformative(const SplitNux* nux) const;
  


  /**
     @brief Dispatches according to terminal/nonterminal state.
   */
  void dispatch(class Frontier* frontier);
  
  /**
     @brief Node-relative reindexing:  indices contiguous on nodes (index sets).
  */
  void reindex(const class BranchSense* branchSense,
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

  
  inline auto getIdxSucc(bool trueBranch) const {
    return trueBranch ? succTrue : succFalse;
  }


  inline auto getSumSucc(bool trueBranch) const {
    return trueBranch ? sumTrue : sum - sumTrue;
  }


  /**
     N.B.:  offset side effected.
   */
  inline auto getOffSucc(bool trueBranch) {
    return trueBranch ? offTrue++ : offFalse++;
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
     @brief Determines whether a given successor is splitable.

     @param sense indicates the branch sense.

     @param[out] succExtent outputs the successor node size.

     @return true iff the successor is live and has sufficient size.
   */
  inline bool succSplitable(bool sense,
			    IndexT& succExtent) const {
    succExtent = getExtentSucc(sense);
    return !succExtinct(sense) && succExtent >= minNode;
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


  /**
     @brief L/R accessor for subtree-relative reindexing.

     @param isExpl is true iff sample index tagged explicit.

     @param explLeft is true iff index both tagged explicitly left.

     @param pathSucc outputs the (possibly pseudo) successor path.

     @param idxSucc outputs the (possibly pseudo) successor index.

     @return index (possibly pseudo) of successor index set.
   */
  inline IndexT offspring(const class BranchSense* branchSense,
                          IndexT sIdx,
                          IndexT& pathSucc,
                          IndexT& ptSucc) {
    return doesSplit ? offspringLive(branchSense->senseTrue(sIdx, !trueEncoding), pathSucc, ptSucc) : offspringTerm(pathSucc, ptSucc);
  }

  
  /**
     @brief Set path and pretree successor of nonterminal.

     @param trueBranch indicates branch sense.

     @return (possibly psuedo-) index of successor IndexSet.
   */
  inline IndexT offspringLive(bool trueBranch,
                          IndexT& pathSucc,
                          IndexT& ptSucc) {
      pathSucc = getPathSucc(trueBranch);
      ptSucc = getPTSucc(trueBranch);
      return getIdxSucc(trueBranch);
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
  inline IndexT offspring(const class BranchSense* branchSense,
                          IndexT sIdx,
                          unsigned int& pathSucc,
                          IndexT& idxSucc,
                          IndexT& ptSucc) {
    if (doesSplit) {
      bool trueBranch = branchSense->senseTrue(sIdx, !trueEncoding);
      idxSucc = getOffSucc(trueBranch);
      return offspringLive(trueBranch, pathSucc, ptSucc);
    }
    else {
      idxSucc = offOnly++;
      return offspringTerm(pathSucc, ptSucc);
    }
  }
};

#endif
