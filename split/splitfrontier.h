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
#include "sumcount.h"
#include "algparam.h"

#include <vector>

enum class EncodingStyle { direct, trueBranch };


/**
   @brief Enapsulates contributions of an individual split to frontier.
 */
struct CritEncoding {
  double sum; // sum of responses over encoding.
  IndexT sCount; // # samples encoded.
  IndexT extent; // # SR indices encoded.
  vector<SumCount> scCtg; // Response sum decomposed by category.
  const IndexT implicitTrue; // # implicit SR indices.
  const bool increment; // True iff encoding is additive else subtractive.
  const bool exclusive; // True iff update is masked.
  const EncodingStyle style; // Whether direct or true-branch.
  
  CritEncoding(const class SplitFrontier* frontier,
	       const class SplitNux* nux,
	       PredictorT nCtg,
	       bool excl,
	       bool incr = true);


  ~CritEncoding() {}

  inline bool trueEncoding() const {
    return implicitTrue == 0;
  };

  /**
     @brief Sample count getter.
   */
  IndexT getSCountTrue(const class SplitNux* nux) const;


  /**
     @return sum of responses contributing to true branch.
   */
  double getSumTrue(const class SplitNux* nux) const;


  /**
     @return # SR inidices contributing to true branch.
   */
  IndexT getExtentTrue(const class SplitNux* nux) const;
  

  /**
     @brief Accumulates encoding statistics for a single SR index.
   */
  inline void accum(double ySum,
		    IndexT sCount,
		    PredictorT ctg) {
    this->sum += ySum;
    this->sCount += sCount;
    extent++;
    if (!scCtg.empty()) {
      scCtg[ctg] += SumCount(ySum, sCount);
    }
  }


  void getISetVals(const class SplitNux* nux,
		   IndexT& sCountTrue,
		   double& sumTrue,
		   IndexT& extentTrue) const;

  
private:  
  /**
     @brief Outputs the internal contents.
   */
  void accumDirect(IndexT& sCountTrue,
		   double& sumTrue,
		   IndexT& extentTrue) const;


  /**
     @brief Outputs the contributions to the true branch.
   */
  void accumTrue(const class SplitNux* nux,
		 IndexT& sCountTrue,
		 double& sumTrue,
		 IndexT& extentTrue) const ;
};


enum class SplitStyle { slots, bits, topSlot };

/**
   @brief Per-predictor splitting facilities.
 */
// Currently implemented in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitFrontier {
  void setPrebias();

  void consumeFrontier(class PreTree* pretree);


protected:
  const class SummaryFrame* frame; // Summarizes the internal predictor reordering.
  const class RankedFrame* rankedFrame; // Represents observations as RLEs.
  class Frontier* frontier;  // Current frontier of the partition tree.
  const PredictorT nPred;
  unique_ptr<class ObsPart> obsPart;  // Partitioned observatsion.
  bool compoundCriteria; // True iff criteria may be multiple-valued.
  EncodingStyle encodingStyle; // How to update observation tree.

  IndexT nSplit; // # subtree nodes at current layer.
  unique_ptr<class RunSet> runSet; // Run accumulators for the current frontier.
  unique_ptr<class CutSet> cutSet; // Cut accumulators for the current frontier.
  
  vector<double> prebias; // Initial information threshold.

  vector<PreCand> restageCand;

  // Per-split accessors for candidate vector.  Reset by DefMap.
  vector<IndexT> candOff;  // Lead candidate position:  cumulative
  vector<PredictorT> nCand;  // At most nPred etries per candidate.

  vector<vector<class SplitNux>> nuxCompound; // Used iff compound criteria enabled.

  vector<class SplitNux> nuxMax;  // Used iff simple criteria enabled.

  vector<class SplitNux> maxCandidates(const vector<class SplitNux>& sc);
  
  class SplitNux candMax(const vector<class SplitNux>& sc,
			 IndexT splitOff,
			 IndexT nSplitFrontier) const;

  
  void consumeSimple(const vector<class SplitNux>& critSimple,
		     PreTree* pretree) const;

  
  void consumeCompound(const vector<vector<class SplitNux>>& critCompound,
		       PreTree* preTre) const;

  /**
     @brief Consumes each criterion in the vector.
   */
  void consumeCriteria(class PreTree* pretree,
		       const vector<class SplitNux>& nuxCrit) const;
  
  /**
     @brief Retrieves the type-relative index of a numerical predictor.

     @param predIdx is the predictor index.

     @return placement-adjusted index.
   */
  PredictorT getNumIdx(PredictorT predIdx) const;


  /**
     @brief Identifies splitable coordinates.

     @return splitNux candidates corresponding to splitable coordinates.
  */
  virtual vector<SplitNux> postSchedule(class DefMap* defMap,
					vector<PreCand>& preCand);

  
  /**
     @brief Dispatches splitting criterion to pretree according to predictor type.
   */
  void consumeCriterion(class PreTree* pretree,
			const class SplitNux* nux) const;


public:

  SplitFrontier(const class SummaryFrame* frame_,
                class Frontier* frontier_,
                const class Sample* sample,
		bool compoundCriteria_,
		EncodingStyle encodingStyle_);


  auto getEncodingStyle() const {
    return encodingStyle;
  }
  

  /**
     @brief Passes ObsPart through to Sample method.
   */
  vector<struct StageCount> stage(const class Sample* sample);


  /**
     @brief Builds an accumulator of the appropriate type.

     @param cand is the candidate associated to the accumulator.
   */
  IndexT addAccumulator(const class SplitNux* cand,
			PredictorT runCount) const;


  /**
     @brief Records splitting state associated with cut.
   */
  void writeCut(const class SplitNux* nux,
		const class CutAccum* accum) const;


  /**
     @brief Instructs (argmax) candidate to update its members.
   */
  void accumUpdate(const class SplitNux* cand) const;
  

  /**
     @brief Updates branch sense vector according to split specification.

     @brief[in, out] branchSense maps sample indices to their branch sense.

     @param[in, out] enc accumulates the splitting statistics.

     @param topOnly requests only the topmost range.

     @parm range is the SR range of the split, if specified.
   */
  CritEncoding nuxEncode(const class SplitNux* nux,
			 class BranchSense* branchSense,
			 const IndexRange& range = IndexRange(),
			 bool increment = true) const;



  void encodeCriterion(class IndexSet* iSet,
		       class SplitNux* nuxMax,
		       class BranchSense* branchSense) const;


  /**
     @brief Computes cut-based SR index range for numeric splits.
   */
  IndexRange getCutRange(const class SplitNux* nux,
			 const class CritEncoding& enc) const;

  
  /**
     @brief Updates observation partition and splits.

     Main entry.
   */
  void restageAndSplit(vector<class IndexSet>& indexSet,
		       class DefMap* defMap,
		       class BranchSense* branchSense,
		       class PreTree* pretree);


  vector<PreCand>& getRestageCand() {
    return restageCand;
  }

  
  /**
     @brief Updates the data (observation) partition.
   */
  void restage(const class DefMap* defMap);


  /**
     @brief Pass-through to data partition method.

     @param cand is a splitting candidate.

     @return pointer to beginning of partition associated with the candidate.
   */
  class SampleRank* getPredBase(const SplitNux* cand) const;

  
  /**
     @brief Interpolates a cutting quantile according to front-end specification.

     @return interpolated quantile for cut.
   */
  double getQuantRank(const class SplitNux* nux) const;


  /**
     @brief Gets right SR index of cut.
   */
  IndexT getIdxRight(const class SplitNux* nux) const;

  
  /**
     @brief Get left SR index of cut.
   */
  IndexT getIdxLeft(const class SplitNux* nux) const;


  /**
     @return count of implicit SR indices targeted to true branch.
   */
  IndexT getImplicitTrue(const class SplitNux* nux) const;


  /**
     @return true iff split is a left cut.
   */
  bool leftCut(const SplitNux* cand) const;
  
  
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
  bool isFactor(const class SplitNux* nux) const;


  /**
     @brief Getter for pre-bias value, by index.

     @param splitIdx is the index.

     @param return pre-bias value.
   */
  inline double getPrebias(const PreCand& preCand) const {
    return prebias[preCand.splitCoord.nodeIdx];
  }


  inline PredictorT getNPred() const {
    return nPred;
  }


  /**
     @brief Getter for split count.
   */
  inline IndexT getNSplit() const {
    return nSplit;
  }


  /**
     @brief Getter for induced pretree index.
   */
  IndexT getPTId(const PreCand& preCand) const;


  /**
     @brief Passes through to Frontier method.

     @return true iff indexed split is not splitable.
   */
  bool isUnsplitable(IndexT splitIdx) const;


  /**
     @brief Pass-through to Frontier getters.
   */
  double getSum(const PreCand& preCand) const;

  
  IndexT getSCount(const PreCand& preCand) const;

  
  /**
     @return SR range of indexed split.
  */
  IndexRange getRange(const class DefMap* defMap,
		      const PreCand& preCand) const; 


  /**
     @brief Passes through to ObsPart method.
   */
  IndexT* getBufferIndex(const class SplitNux* nux) const;
  
  
  /**
   */
  RunAccumT* getRunAccum(PredictorT setIdx) const;


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
     @brief Fixes factor splitting style.
   */
  virtual SplitStyle getFactorStyle() const = 0;
  

  /**
     @brief Passes through to Cand method.
   */
  vector<PreCand> precandidates(const class DefMap* defMap);


  /**
     @brief Dumps run-vector contents for diagnostics.
   */
  struct RunDump dumpRun(PredictorT accumIdx) const;

  
  /**
     @brief Builds compressed set of indices for candidate vector.
   */
  void setOffsets(const vector<class SplitNux>& sched);
  

  virtual ~SplitFrontier();


  virtual void layerPreset() = 0;


  virtual void setPrebias(IndexT splitIdx,
                          double sum,
                          IndexT sCount) = 0;

  
  /**
     @brief Clears per-frontier vectors.

     TODO:  Allocate new frontiers and deprecate persistance.
   */
  virtual void clear();
};


struct SFReg : public SplitFrontier {
  // Bridge-supplied monotone constraints.  Length is # numeric predictors
  // or zero, if none so constrained.
  static vector<double> mono;

  // Per-layer vector of uniform variates.
  vector<double> ruMono;

  SFReg(const class SummaryFrame* frame,
	class Frontier* frontier,
	const class Sample* sample,
	bool compoundCriteria,
	EncodingStyle encodingStyle);

  ~SFReg();

  /**
     @brief Caches a dense local copy of the mono[] vector.

     @param summaryFrame contains the predictor block mappings.

     @param bridgeMono has length equal to the predictor count.  Only
     numeric predictors may have nonzero entries.
  */
  static void immutables(const class SummaryFrame* summaryFrame,
                         const vector<double>& feMono);

  /**
     @brief Resets the monotone constraint vector.
   */
  static void deImmutables();
  

  /**
     @brief Determines whether a regression pair undergoes constrained splitting.

     @return constraint sign, if within the splitting probability, else zero.
  */
  int getMonoMode(const class SplitNux* cand) const;

  
  /**
     @brief Sets layer-specific values for the subclass.
  */
  void layerPreset();
};


class SFCtg : public SplitFrontier {
protected:
  vector<double> sumSquares; // Per-layer sum of squares, by split.
  vector<double> ctgSumAccum; // Numeric predictors:  accumulate sums.


  const PredictorT nCtg;
  vector<vector<double> > ctgSum; // Per-category response sums, by node.

public:
  SFCtg(const class SummaryFrame* frame,
	class Frontier* frontier,
	const class Sample* sample,
	bool compoundCriteria,
	EncodingStyle encodingStyle,
	PredictorT nCtg_);


  /**
     @brief Getter for training response cardinality.

     @return nCtg value.
   */
  inline PredictorT getNCtg() const {
    return nCtg;
  }


  /**
     @brief Accesses per-category sum vector associated with candidate's node.

     @param cand is the splitting candidate.

     @return reference vector of per-category sums.
   */
  const vector<double>& getSumSlice(const class SplitNux* cand) const;


  /**
     @brief Provides slice into accumulation vector for a splitting candidate.

     @param cand is the splitting candidate.

     @return raw pointer to per-category accumulation vector for pair.
   */
  double* getAccumSlice(const class SplitNux* cand);


  /**
     @brief Per-node accessor for sum of response squares.

     @param cand is a splitting candidate.

     @return sum, over categories, of node reponse values.
   */
  double getSumSquares(const class SplitNux* cand) const;
};


#endif
