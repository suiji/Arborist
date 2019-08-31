// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CART_SPLITFRONTIER_H
#define CART_SPLITFRONTIER_H

/**
   @file splitnode.h

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
  
 protected:
  const class SummaryFrame* frame;
  const class RankedFrame* rankedFrame;
  class Frontier* frontier;
  IndexT bagCount; 
  const IndexT noSet; // Unreachable setIdx for SplitCand.
  unique_ptr<class ObsPart> obsPart;
  IndexT splitCount; // # subtree nodes at current level.
  unique_ptr<class Run> run; // Run sets for the current level.
  vector<class SplitCand> splitCand; // Schedule of splits.

  vector<double> prebias; // Initial information threshold.
  // Per-split accessors for candidate vector.  Set to splitCount
  // and cleared after use:
  vector<unsigned int> candOff;  // Lead candidate position.
  vector<unsigned int> nCand;  // Number of candidates.

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


  void scheduleSplits(//const class Frontier *index,
		      const class Level *levelFront);

  /**
     @brief Emplaces new candidate with specified coordinates.
   */
  IndexT preschedule(//const Frontier* index,
                     const SplitCoord& splitCoord,
                     unsigned int bufIdx);

  /**
     @brief Passes through to ObsPart method.
   */
  double blockReplay(class IndexSet* iSet,
                     const IndexRange& range,
                     bool leftExpl,
                     class BV* replayExpl,
                     class BV* replayLeft,
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
     @brief Pass-through to row-rank method.

     @param cand is the candidate.

     @return rank of dense value, if candidate's predictor has one.
   */
  unsigned int getDenseRank(const SplitCand* cand) const;

  
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
  IndexT getLHExtent(const class IndexSet& iSet) const;

  IndexT getPredIdx(const class IndexSet* iSet) const;

  unsigned int getBufIdx(const class IndexSet* iSet) const;

  PredictorT getCardinality(const class IndexSet* iSet) const;

  double getInfo(const class IndexSet* iSet) const;

  IndexRange getExplicitRange(const class IndexSet* iSet) const;

  IndexRange getRankRange(const class IndexSet* iSet) const;

  bool leftIsExplicit(const class IndexSet* iSet) const;

  unsigned int getSetIdx(const class IndexSet* iSet) const;


  SplitSurvey consume(class PreTree* pretree,
                      vector<class IndexSet>& indexSet,
                      class BV* replayExpl,
                      class BV* replayLeft);

  
  void consume(class PreTree* pretree,
               class IndexSet& iSet,
               class BV* replayExpl,
               class BV* replayLeft,
               SplitSurvey& survey) const;

  
  /**
     @brief Passes through to run member.

     @return true iff split is left-explicit
   */
  void branch(class PreTree* pretree,
              class IndexSet* iSet,
              class BV* bvLeft,
              class BV* bvRight) const;


  /**
     @brief Replays run-based criterion and updates pretree.
   */
  void critRun(class PreTree* pretree,
               class IndexSet* iSet,
               class BV* bvLeft,
               class BV* bvRight) const;

  /**
     @brief Replays cut-based criterion and updates pretree.
   */
  void critCut(class PreTree* pretree,
               class IndexSet* iSet,
               BV* bvLeft,
               BV* bvRight) const;


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

  
  /**
     @brief Invokes algorithm-specific splitting methods.
   */
  void split();


  vector<class SplitNux> maxCandidates();
  
  class SplitNux maxSplit(unsigned int splitOff,
                          unsigned int nSplitFrontier) const;
  
  virtual void splitCandidates() = 0;
  virtual ~SplitFrontier();
  virtual void setRunOffsets(const vector<unsigned int> &safeCounts) = 0;
  virtual void levelPreset() = 0;

  virtual void setPrebias(IndexT splitIdx,
                          double sum,
                          IndexT sCount) = 0;

  virtual void clear();
};


/**
   @brief Splitting facilities specific regression trees.
 */
class SFReg : public SplitFrontier {
  // Bridge-supplied monotone constraints.  Length is # numeric predictors
  // or zero, if none so constrained.
  static vector<double> mono;

  // Per-level vector of uniform variates.
  vector<double> ruMono;

  void splitCandidates();

 public:

  /**
     @brief Caches a dense local copy of the mono[] vector.

     @param frameMap contains the predictor block mappings.

     @param bridgeMono has length equal to the predictor count.  Only
     numeric predictors may have nonzero entries.
  */
  static void Immutables(const class SummaryFrame* frame,
                         const vector<double> &feMono);

  /**
     @brief Resets the monotone constraint vector.
   */
  static void DeImmutables();
  
  SFReg(const class SummaryFrame* frame_,
        class Frontier* frontier_,
	const class Sample* sample);

  ~SFReg();
  void setRunOffsets(const vector<unsigned int> &safeCount);
  void levelPreset();
  void clear();

  /**
     @brief Determines whether a regression pair undergoes constrained splitting.
     @return The sign of the constraint, if within the splitting probability, else zero.
*/
  int getMonoMode(const class SplitCand* cand) const;

  /**
     @brief Weighted-variance pre-bias computation for regression response.

     @param sum is the sum of samples subsumed by the index node.

     @param sCount is the number of samples subsumed by the index node.

     @return sum squared, divided by sample count.
  */
  inline void setPrebias(IndexT splitIdx,
			 double sum,
			 IndexT sCount) {
    prebias[splitIdx] = (sum * sum) / sCount;
  }

};


/**
   @brief Splitting facilities for categorical trees.
 */
class SFCtg : public SplitFrontier {
// Numerical tolerances taken from A. Liaw's code:
  static constexpr double minDenom = 1.0e-5;
  static constexpr double minSumL = 1.0e-8;
  static constexpr double minSumR = 1.0e-5;

  const PredictorT nCtg; // Response cardinality.
  vector<double> sumSquares; // Per-level sum of squares, by split.
  vector<double> ctgSumAccum; // Numeric predictors:  accumulate sums.

  /**
     @brief Initializes per-level sum and FacRun vectors.
  */
  void levelPreset();

  /**
     @brief Clears summary state associated with this level.
   */
  void clear();

  /**
     @brief Collects splitable candidates from among all restaged cells.
   */
  void splitCandidates();

  /**
     @brief RunSet initialization utitlity.

     @param safeCount gives a conservative per-predictor count of distinct runs.
   */
  void setRunOffsets(const vector<unsigned int> &safeCount);


  /**
     @brief Initializes numerical sum accumulator for currently level.

     @parm nPredNum is the number of numerical predictors.
   */
  void levelInitSumR(PredictorT nPredNum);

  
  /**
     @brief Gini pre-bias computation for categorical response.

     @param splitIdx is the level-relative node index.

     @param sum is the sum of samples subsumed by the index node.

     @param sCount is the number of samples subsumed by the index node.

     @return sum of squares divided by sum.
  */
  inline void setPrebias(IndexT splitIdx,
                         double sum,
                         IndexT sCount) {
    prebias[splitIdx] = sumSquares[splitIdx] / sum;
  }


 public:
  vector<vector<double> > ctgSum; // Per-category response sums, by node.

  SFCtg(const class SummaryFrame* frame_,
        class Frontier* frontier_,
	const class Sample* sample,
	PredictorT nCtg_);
  ~SFCtg();


  /**
     @brief Getter for training response cardinality.

     @return nCtg value.
   */
  inline PredictorT getNCtg() const {
    return nCtg;
  }

  
  /**
     @brief Determine whether an ordered pair of sums is acceptably stable
     to appear in the denominator.

     Only relevant for instances of extreme case weighting.  Currently unused
     and may be obsolete.

     @param sumL is the left-hand sum.

     @param sumR is the right-hand sum.

     @return true iff both sums suitably stable.
   */
  inline bool stableSum(double sumL, double sumR) const {
    return sumL > minSumL && sumR > minSumR;
  }


  /**
     @brief Determines whether a pair of sums is acceptably stable to appear
     in the denominators.

     Only relevant for instances of extreme case weighting.  Currently unused
     and may not be useful if training responses are normalized.

     @param sumL is the left-hand sum.

     @param sumR is the right-hand sum.

     @return true iff both sums suitably stable.
   */
  inline bool stableDenom(double sumL, double sumR) const {
    return sumL > minDenom && sumR > minDenom;
  }
  

  /**
     @brief Accesses per-category sum vector associated with candidate's node.

     @param cand is the splitting candidate.

     @return reference vector of per-category sums.
   */
  const vector<double>& getSumSlice(const class SplitCand* cand);


  /**
     @brief Provides slice into accumulation vector for a splitting candidate.

     @param cand is the splitting candidate.

     @return raw pointer to per-category accumulation vector for pair.
   */
  double* getAccumSlice(const class SplitCand* cand);


  /**
     @brief Per-node accessor for sum of response squares.

     @param cand is a splitting candidate.

     @return sum, over categories, of node reponse values.
   */
  double getSumSquares(const class SplitCand *cand) const;
};


#endif
