// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CART_SPLITFRONTIER_H
#define CART_SPLITFRONTIER_H

/**
   @file splitfrontier.h

   @brief Manages node splitting across the tree frontier, by response type.

   @author Mark Seligman

 */

#include "splitcoord.h"
#include "typeparam.h"
#include "sumcount.h"

#include <vector>

/**
   @brief Enumerates split characteristics over a trained frontier.
 */
struct SplitSurvey {
  IndexT leafCount; // Number of terminals in this level.
  IndexT idxLive; // Extent of live buffer indices.
  IndexT splitNext; // Number of splitable nodes in next level.
  IndexT idxMax; // Maximum index.

  SplitSurvey() :
    leafCount(0),
    idxLive(0),
    splitNext(0),
    idxMax(0){
  }


  /**
     brief Imputes the number of successor nodes, including pseudosplits.
   */
  IndexT succCount(IndexT splitCount) const {
    IndexT leafNext = 2 * (splitCount - leafCount) - splitNext;
    return splitNext + leafNext + leafCount;
  }
};


/**
   @brief Per-predictor splitting facilities.
 */
// Currently implemented in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitFrontier {
  vector<class SplitNux> nuxMax; // Rewritten following each splitting event.
  void setPrebias();//class Frontier *index);
  vector<IndexT> candOffset; // Offset indices for each scheduled candidate.

protected:
  const class SummaryFrame* frame;
  const class RankedFrame* rankedFrame;
  class Frontier* frontier;
  const PredictorT nPred;
  const IndexT noSet; // Unreachable setIdx for SplitNux.
  unique_ptr<class ObsPart> obsPart;
  IndexT splitCount; // # subtree nodes at current level.
  unique_ptr<class Run> run; // Run sets for the current level.
  vector<class SplitNux> splitCand; // Schedule of splits.

  vector<double> prebias; // Initial information threshold.
  // Per-split accessors for candidate vector.  Set to splitCount
  // and cleared after use:
  vector<IndexT> candOff;  // Lead candidate position.
  vector<IndexT> nCand;  // Number of candidates.

  /**
     @brief Retrieves the type-relative index of a numerical predictor.

     @param predIdx is the predictor index.

     @return placement-adjusted index.
   */
  PredictorT getNumIdx(PredictorT predIdx) const;


public:

  
  SplitFrontier(const class SummaryFrame *frame_,
                class Frontier* frontier_,
                const class Sample* sample);

  void
  cacheOffsets(vector<IndexT>& candOffset);

  
  void
  scheduleSplits(const class Bottom* bottom);

  
  /**
     @brief Emplaces new candidate with specified coordinates.
   */
  IndexT preschedule(const SplitCoord& splitCoord,
                     unsigned int bufIdx);

  /**
     @brief Passes through to ObsPart method.
   */
  double blockReplay(class IndexSet* iSet,
                     const IndexRange& range,
                     bool leftExpl,
		     class Replay* replay,
                     vector<SumCount>& ctgCrit) const;

  /**
     @brief Passes ObsPart through to Sample method.
   */
  vector<class StageCount> stage(const class Sample* sample);
  
  
  /**
     @brief Passes through to ObsPart method.
   */
  void restage(class Level* levelFrom,
               class Level* levelTo,
               const SplitCoord& splitCoord,
               unsigned int bufIdx) const;


  /**
     @brief Pass-through to data partition method.

     @param cand is a splitting candidate.

     @return pointer to beginning of partition associated with the candidate.
   */
  class SampleRank* getPredBase(const SplitNux* cand) const;

  
  /**
     @brief Pass-through to row-rank method.

     @param cand is the candidate.

     @return rank of dense value, if candidate's predictor has one.
   */
  IndexT getDenseRank(const SplitNux* cand) const;

  
  /**
     @brief Pass-through to frame-map method.

     @param predIdx is a predictor index.

     @return true iff predictor is a factor.
   */
/**
   @brief Determines whether split coordinate references a factor value.

   @param splitCoord is the split coordinate.

   @return true iff predictor is a factor.
 */
  bool isFactor(const SplitCoord& splitCoord) const;


  /**
     @brief Collects nonterminal parameters from nux and passes to index set.

     @param iSet is the index set absorbing the split parameters.
   */
  void consumeCriterion(class IndexSet* iSet) const;

  void nonterminal(const class IndexSet* iSet,
                   double& minInfo,
                   IndexT& lhsCount,
                   IndexT& lhExtent) const;
  
  /**
     @brief Determines whether a potential split is sufficiently informative.

     @param splitIdx is the split position.

     @bool true iff threshold exceeded.
   */
  bool isInformative(const class IndexSet* iSet) const;


  /**
     @brief Gives the extent of one a split's descendants.

     Which descendant must not be relevant to the caller.
     
     @param splitIdx is the split position.

     @return descendant extent.
   */
  IndexT getLHExtent(const class IndexSet* iSet) const;

  IndexT getPredIdx(const class IndexSet* iSet) const;

  unsigned int getBufIdx(const class IndexSet* iSet) const;

  
  PredictorT getCardinality(const class IndexSet* iSet) const;

  
  double getInfo(const class IndexSet* iSet) const;

  IndexRange getExplicitRange(const class IndexSet* iSet) const;

  IndexRange getRankRange(const class IndexSet* iSet) const;

  bool leftIsExplicit(const class IndexSet* iSet) const;

  IndexT getSetIdx(const class IndexSet* iSet) const;


  SplitSurvey consume(class PreTree* pretree,
                      vector<class IndexSet>& indexSet,
                      class Replay* replay);

  
  void consume(class PreTree* pretree,
               class IndexSet& iSet,
               class Replay* replay,
               SplitSurvey& survey) const;

  
  /**
     @brief Passes through to run member.

     @return true iff split is left-explicit
   */
  void branch(class PreTree* pretree,
              class IndexSet* iSet,
	      class Replay* replay) const;


  /**
     @brief Replays run-based criterion and updates pretree.
   */
  void critRun(class PreTree* pretree,
               class IndexSet* iSet,
	       class Replay* replay) const;

  /**
     @brief Replays cut-based criterion and updates pretree.
   */
  void critCut(class PreTree* pretree,
               class IndexSet* iSet,
	       class Replay* replay) const;


  /**
     @brief Getter for pre-bias value, by index.

     @param splitIdx is the index.

     @param return pre-bias value.
   */
  inline double getPrebias(const SplitCoord& splitCoord) const {
    return prebias[splitCoord.nodeIdx];
  }


  /**
   */
  class RunSet *rSet(unsigned int setIdx) const;

  /**
     @brief Initializes state associated with current level.
   */
  void init();

  vector<class SplitNux> maxCandidates();
  
  class SplitNux maxSplit(IndexT splitOff,
                          IndexT nSplitFrontier) const;

  /**
     @brief Invokes algorithm-specific splitting methods.
   */
  void splitCandidates();

  /**
     @brief Determines splitting candidates.
   */
  virtual void
  candidates(const class Frontier* frontier,
	     const class Bottom* bottom) = 0;
  
  virtual void split(class SplitNux* cand) = 0;
  virtual ~SplitFrontier();
  virtual void setRunOffsets(const vector<unsigned int>& safeCounts) = 0;
  virtual void levelPreset() = 0;

  virtual void setPrebias(IndexT splitIdx,
                          double sum,
                          IndexT sCount) = 0;

  virtual void clear();
};


#endif
