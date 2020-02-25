// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_SPLITFRONTIER_H
#define SPLIT_SPLITFRONTIER_H

/**
   @file splitfrontier.h

   @brief Manages node splitting across the tree frontier, by response type.

   @author Mark Seligman

 */

#include "splitcoord.h"
#include "pretree.h"
#include "sumcount.h"

#include <vector>

/**
   @brief Per-predictor splitting facilities.
 */
// Currently implemented in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitFrontier {
  void setPrebias();

  void consumeFrontier(class PreTree* pretree);

  
  void runReplay(class SplitNux* nux,
		 class BranchSense* branchSense,
		 vector<SumCount>& ctgCrit,
		 bool exculsive = false) const;

  void rangeReplay(class SplitNux* nux,
		   const IndexRange& range,
		   class BranchSense* branchSense,
		   vector<SumCount>& ctgCrit) const;
  /**
     @brief As above, but does not assume observations have been restaged.

     @param branchSense modified by exclusive or.
   */
  void rangeReplayExcl(class SplitNux* nux,
		       class BranchSense* branchSense,
		       vector<SumCount>& ctgCrit) const;
  



protected:
  const class SummaryFrame* frame;
  const class RankedFrame* rankedFrame;
  class Frontier* frontier;
  const PredictorT nPred;
  unique_ptr<class ObsPart> obsPart;
  IndexT nSplit; // # subtree nodes at current layer.
  unique_ptr<class Run> run; // Run sets for the current layer.
  
  vector<double> prebias; // Initial information threshold.

  vector<DefCoord> restageCoord;

  // Per-split accessors for candidate vector.  Reset by DefMap.
  vector<IndexT> candOff;  // Lead candidate position:  cumulative
  vector<PredictorT> nCand;  // At most nPred etries per candidate.

  /**
     @brief Retrieves the type-relative index of a numerical predictor.

     @param predIdx is the predictor index.

     @return placement-adjusted index.
   */
  PredictorT getNumIdx(PredictorT predIdx) const;


  /**
     @brief Walks the list of split candidates and invalidates those which
     restaging has marked unsplitable as well as singletons persisting since
     initialization or as a result of bagging.  Fills in run counts, which
     values restaging has established precisely.
  */
  vector<SplitNux>
  postSchedule(class DefMap* defMap,
	       vector<DefCoord>& preCand);


  void
  postSchedule(const DefMap* defMap,
	       const DefCoord& preCand,
	       vector<PredictorT>& runCount,
	       vector<PredictorT>& nCand,
	       vector<SplitNux>& postCand) const;

  
  /**
     @brief Looks up the run count associated with a given node, predictor pair.
     
     @param splitCoord specifies the node, predictor candidate pair.

     @return run count associated with the node, if factor, otherwise zero.
   */
  PredictorT getSetIdx(PredictorT rCount,
		       vector<PredictorT>& runCount) const;

  
  /**
     @brief Dispatches splitting criterion to pretree according to predictor type.
   */
  void consumeCriterion(class PreTree* pretree,
			const class SplitNux* nux) const;


public:

  SplitFrontier(const class SummaryFrame* frame_,
                class Frontier* frontier_,
                const class Sample* sample);
  
  void
  preschedule(const DefCoord& defCoord,
	      vector<DefCoord>& preCand) const;


  /**
     @brief Passes ObsPart through to Sample method.
   */
  vector<struct StageCount> stage(const class Sample* sample);


  /**
     @brief Replays true/false branch sense vector according to SplitNux contents.

     @param exclusive is true iff branch sense to be exclusive-or'ed, otherwise or'ed.
   */
  void nuxReplay(class SplitNux* nux,
		 class BranchSense* branchSense,
		 vector<SumCount>& ctgCrit,
		 bool exclusive = false) const;
  

  /**
     @brief Main entry from frontier loop.
   */
  void restageAndSplit(vector<class IndexSet>& indexSet,
		       class DefMap* defMap,
		       class BranchSense* branchSense,
		       class PreTree* pretree);
  
  /**
     @brief Passes through to ObsPart method.
   */
  void scheduleRestage(const DefCoord& mrra) {
    restageCoord.push_back(mrra);
  }

  /**
     @brief Passes through to RunSet counterpart.
   */
  void lHBits(SplitNux* nux,
	      PredictorT lhBits) const;

  
  /**
     @brief Passes through to RunSet counterpart.
   */
  void lHSlots(SplitNux* nux,
	       PredictorT cutSlot) const;


  void appendSlot(class SplitNux* nux) const;

  
  void restage(const class DefMap* defMap);

  

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


  virtual void consumeNodes(PreTree* pretree) const = 0;


  /**
     @brief Getter for pre-bias value, by index.

     @param splitIdx is the index.

     @param return pre-bias value.
   */
  inline double getPrebias(const SplitCoord& splitCoord) const {
    return prebias[splitCoord.nodeIdx];
  }


  inline PredictorT getNPred() const {
    return nPred;
  }


  inline IndexT getNSplit() const {
    return nSplit;
  }


  IndexT getPTId(const SplitCoord& splitCoord) const;


  /**
     @return unreachable run-set index.
   */
  IndexT getNoSet() const;


  /**
     @brief Passes through to Frontier method.

     @return true iff indexed split is not splitable.
   */
  bool isUnsplitable(IndexT splitIdx) const;


  /**
     @brief Pass-through to Frontier getters.
   */
  double getSum(const SplitCoord& splitCoord) const;

  IndexT getSCount(const SplitCoord& splitCoord) const;

  /**
     @return buffer range of indexed split.
  */
  IndexRange getBufRange(const DefCoord& preCand) const; 


  /**
     @brief Passes through to ObsPart method.
   */
  IndexT* getBufferIndex(const class SplitNux* nux) const;
  
  
  /**
   */
  class RunSet* getRunSet(PredictorT setIdx) const;


  IndexRange getRunBounds(const class SplitNux* nux,
			  PredictorT slot) const;
  
  /**
     @brief Initializes state associated with current layer.
   */
  void init(class BranchSense* branchSense);

  
  /**
     @brief Invokes algorithm-specific splitting methods.
   */
  virtual void split(vector<class IndexSet>& indexSet,
		     vector<class SplitNux>& sc,
		     class BranchSense* branchSense) = 0;


  /**
     @brief Passes through to Cand method.
   */
  vector<DefCoord>
  precandidates(const class DefMap* defMap);

  void setCandOff(const vector<PredictorT>& ncand);
  
  virtual ~SplitFrontier();

  virtual void layerPreset() = 0;

  virtual void setPrebias(IndexT splitIdx,
                          double sum,
                          IndexT sCount) = 0;

  /**
     @brief Clears per-frontier vectors.
   */
  virtual void clear();
};


#endif
