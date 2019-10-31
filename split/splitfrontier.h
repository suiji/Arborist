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
   @brief Enumerates split characteristics over a trained frontier.
 */
struct SplitSurvey {
  IndexT leafCount; // Number of terminals in this layer.
  IndexT idxLive; // Extent of live buffer indices.
  IndexT splitNext; // Number of splitable nodes in next layer.
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
  void setPrebias();

protected:
  vector<unique_ptr<class SplitNux> > nuxMax; // Rewritten following each splitting event.
  const class Cand* cand;
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



public:

  SplitFrontier(const class Cand* cand_,
		const class SummaryFrame* frame_,
                class Frontier* frontier_,
                const class Sample* sample);


  void
  preschedule(const DefCoord& defCoord,
	      vector<DefCoord>& preCand) const;

  double blockReplay(class IndexSet* iSet,
                     const IndexRange& range,
                     bool leftExpl,
		     class Replay* replay,
                     vector<SumCount>& ctgCrit) const;

  /**
     @brief Passes ObsPart through to Sample method.
   */
  vector<struct StageCount> stage(const class Sample* sample);


  /**
     @brief Main entry from frontier loop.
   */
  void restageAndSplit(class DefMap* defMap);
  
  /**
     @brief Passes through to ObsPart method.
   */
  void scheduleRestage(const DefCoord& mrra) {
    restageCoord.push_back(mrra);
  }

  /**
     @brief Passes through to RunSet method.

     @param setIdx is the Runset index.
   */
  IndexT lHBits(PredictorT setIdx,
		PredictorT lhBits,
		IndexT& lhSCount) const;

  
  IndexT lHSlots(PredictorT setIdx,
		 PredictorT cutSlot,
		 IndexT& lhSCount) const;
  
  
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

  DefCoord getDefCoord(const class IndexSet* iSet) const;
  
  
  PredictorT getCardinality(const class IndexSet* iSet) const;

  
  double getInfo(const class IndexSet* iSet) const;

  IndexRange getExplicitRange(const class IndexSet* iSet) const;

  double getQuantRank(const class IndexSet* iSet) const;

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


  inline PredictorT getNPred() const {
    return nPred;
  }


  inline IndexT getNSplit() const {
    return nSplit;
  }


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
   */
  class RunSet *rSet(unsigned int setIdx) const;

  /**
     @brief Initializes state associated with current layer.
   */
  void init();

  vector<unique_ptr<class SplitNux> >
  maxCandidates(const vector<class SplitNux>& sc);
  
  unique_ptr<class SplitNux>
  maxSplit(const vector<class SplitNux>& sc,
			  IndexT splitOff,
                          IndexT nSplitFrontier) const;

  /**
     @brief Invokes algorithm-specific splitting methods.
   */
  virtual void
  split(vector<class SplitNux>& sc) = 0;


  /**
     @brief Passes through to Cand method.
   */
  vector<DefCoord>
  precandidates(const class DefMap* defMap);

  void setCandOff(const vector<PredictorT>& ncand);
  
  virtual ~SplitFrontier();
  virtual void setRunOffsets(const vector<PredictorT>& safeCounts) = 0;
  virtual void layerPreset() = 0;

  virtual void setPrebias(IndexT splitIdx,
                          double sum,
                          IndexT sCount) = 0;

  virtual void clear();
};


#endif
