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

#include "splitnux.h"
#include "sumcount.h"
#include "algparam.h"
#include "cutfrontier.h"
#include "runfrontier.h"
#include "critencoding.h"

#include <vector>
#include <functional>

enum class SplitStyle { slots, bits, topSlot };


/**
   @brief Per-predictor splitting facilities.
 */
// Currently implemented in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitFrontier {

protected:
  const class PredictorFrame* frame; // Summarizes the internal predictor reordering.
  class Frontier* frontier;  // Current frontier of the partition tree.
  class InterLevel* interLevel;
  const bool compoundCriteria; // True iff criteria may be multiple-valued.
  EncodingStyle encodingStyle; // How to update observation tree.
  const SplitStyle splitStyle;
  const IndexT nSplit; // # subtree nodes at current layer.
  void (SplitFrontier::* splitter)(const CandType& cand,
				   class BranchSense&); // Splitting method.

  unique_ptr<RunSet> runSet; // Run accumulators for the current frontier.
  unique_ptr<CutSet> cutSet; // Cut accumulators for the current frontier.


  /**
     @brief Derives and applies maximal simple criteria.
   */
  void maxSimple(const vector<SplitNux>& sc,
		 class BranchSense& branchSense);

  
  vector<class SplitNux> maxCandidates(const vector<vector<class SplitNux>>& candVV);

  
  /**
     @brief Retrieves the type-relative index of a numerical predictor.

     @param predIdx is the predictor index.

     @return placement-adjusted index.
   */
  PredictorT getNumIdx(PredictorT predIdx) const;


public:

  SplitFrontier(class Frontier* frontier_,
		bool compoundCriteria_,
		EncodingStyle encodingStyle_,
		SplitStyle splitStyle_,
		void (SplitFrontier::* splitter_)(const CandType&,
						  class BranchSense&));

  virtual ~SplitFrontier() = default;
  

  void split(const CandType& cand,
	     class BranchSense& branchSense);


  RunSet* getRunSet() const {
    return runSet.get();
  }
  

  auto getEncodingStyle() const {
    return encodingStyle;
  }

  
  /**
     @return true iff compound criteria are supported.
   */
  bool getCompound() const {
    return compoundCriteria;
  }

  
  /**
     @brief Computes number of bits employed by criterion.

     @return frame run count plus one for proxy bit.
   */
  PredictorT critBitCount(const SplitNux& nux) const;


  /**
     @brief Passes through to RunSet method.

     Sets bit offsets of factors encoding true criterion.
   */
  void setTrueBits(const SplitNux& nux,
		   class BV* splitBits,
		   size_t bitPos) const {
    runSet->setTrueBits(interLevel, nux, splitBits, bitPos);
  }


  /**
     @brief As above, but observed bits.
   */
  void setObservedBits(const SplitNux& nux,
		       class BV* splitBits,
		       size_t bitPos) const {
    runSet->setObservedBits(interLevel, nux, splitBits, bitPos);
  }


  /**
     @brief Increments accumulator in respective set.

     @param cand is the candidate associated to the accumulator.

     @return pre-incremented index value.
   */
  IndexT accumulatorIndex(const class SplitNux& cand) const;


  /**
     @brief Records splitting state associated with cut.
   */
  void writeCut(const class SplitNux& nux,
		const class CutAccum& accum) const;


  /**
     @brief Instructs (argmax) candidate to update its members.
   */
  void accumUpdate(const class SplitNux& cand) const;
  

  vector<IndexRange> getRange(const SplitNux& nux,
			      const CritEncoding& enc) const;


  /**
     @brief Computes cut-based SR index range for numeric splits.
   */
  vector<IndexRange> getCutRange(const class SplitNux& nux,
				 const struct CritEncoding& enc) const;


  /**
     @brief Pass-through to data partition method.

     @param cand is a splitting candidate.

     @return pointer to beginning of partition associated with the candidate.
   */
  class Obs* getPredBase(const SplitNux& cand) const;

  
  /**
     @brief Interpolates a cutting quantile according to front-end specification.

     @return interpolated quantile for cut.
   */
  double getQuantRank(const class SplitNux& nux) const;


  /**
     @brief Gets right SR index of cut.
   */
  IndexT getIdxRight(const class SplitNux& nux) const;

  
  /**
     @brief Get left SR index of cut.
   */
  IndexT getIdxLeft(const class SplitNux& nux) const;


  /**
     @return count of implicit SR indices targeted to true branch.
   */
  IndexT getImplicitTrue(const class SplitNux& nux) const;


  /**
     @return true iff split is a left cut.
   */
  bool leftCut(const SplitNux& cand) const;
  

  /**
     @brief Pass-through to frame-map method.

     @param predIdx is a predictor index.

     @return true iff predictor is a factor.
   */
  bool isFactor(const class SplitNux& nux) const;


  /**
     @brief Getter for split count.
   */
  inline IndexT getNSplit() const {
    return nSplit;
  }


  /**
     @brief Getter for induced pretree index.
   */
  IndexT getPTId(const StagedCell* preCand) const;


  /**
     @brief Pass-through to Frontier getters.
   */
  double getSum(const StagedCell* preCand) const;

  
  IndexT getSCount(const StagedCell* preCand) const;


  /** Pass-throughs to Frontier methods.
   */
  double getSumSucc(const StagedCell* mrra,
		    bool sense) const;

  
  IndexT getSCountSucc(const StagedCell* mrra,
		       bool sense) const;


  /**
     @brief Passes through to ObsPart method.

     @return observation partition.
   */
  const class ObsPart* getPartition() const;
  

  /**
     @brief Passes through to ObsPart method.
   */
  IndexT* getIdxBuffer(const class SplitNux& nux) const;
  

  /**
     @brief Classification sublcasses return # categories; others zero.
   */
  PredictorT getNCtg() const;


  /**
     @brief Updates accumulator state for successful split.

     Side-effects branchOffset.

     @return encoding associated with split.
   */
  CritEncoding splitUpdate(const SplitNux& nux,
			   class BranchSense& branchSense,
			   const IndexRange& range = IndexRange(),
			   bool increment = true) const;

  
  SplitStyle getFactorStyle() const {
    return splitStyle;
  }


  /**
     @brief Separates candidates into split-specific vectors.
   */
  vector<vector<class SplitNux>> groupCand(const vector<SplitNux>& cand) const;

  
  /**
     @brief Presets frontier-wide accumulator state.
   */
  void accumPreset();


  // These are run-time invariant and need not be virtual:
  virtual double getScore(const class IndexSet& iSet) const = 0;
};


struct SFReg : public SplitFrontier {
  // Bridge-supplied monotone constraints.  Length is # numeric predictors
  // or zero, if none so constrained.
  static vector<double> mono;

  // Per-layer vector of uniform variates.
  vector<double> ruMono;

  SFReg(class Frontier* frontier,
	bool compoundCriteria,
	EncodingStyle encodingStyle,
	SplitStyle splitStyle,
	void (SplitFrontier::* splitter_)(const CandType&,
					  class BranchSense&));


  /**
     @brief Caches a dense local copy of the mono[] vector.

     @param summaryFrame contains the predictor block mappings.

     @param bridgeMono has length equal to the predictor count.  Only
     numeric predictors may have nonzero entries.
  */
  static void immutables(const class PredictorFrame* summaryFrame,
                         const vector<double>& feMono);

  /**
     @brief Resets the monotone constraint vector.
   */
  static void deImmutables();
  

  /**
     @brief Determines whether a regression pair undergoes constrained splitting.

     @return constraint sign, if within the splitting probability, else zero.
  */
  int getMonoMode(const class SplitNux& cand) const;

  
  /**
     @brief Sets subclass-specific splitting parameters.
  */
  void monoPreset();

  
  double getScore(const class IndexSet& iSet) const;
};


class SFCtg : public SplitFrontier {
protected:
  const PredictorT nCtg;
  vector<vector<double> > ctgSum; // Per-category response sums, by node.
  vector<double> sumSquares; // Per-layer sum of squares, by split.
  vector<double> ctgJitter; // Breaks scoring ties at node.

  
public:
  SFCtg(class Frontier* frontier,
	bool compoundCriteria,
	EncodingStyle encodingStyle,
	SplitStyle splitStyle,
	void (SplitFrontier::* splitter_) (const CandType&,
					   class BranchSense&));
  
  double getScore(const class IndexSet& iSet) const;


  /**
     @brief Copies per-category sum vector associated with candidate's node.

     @param cand is the splitting candidate.

     @return reference to vector of per-category sums.
   */
  const vector<double>& ctgNodeSums(const class SplitNux& cand) const;


  /**
     @brief Per-node accessor for sum of response squares.

     @param cand is a splitting candidate.

     @return sum, over categories, of node reponse values.
   */
  double getSumSquares(const class SplitNux& cand) const;
};


#endif
