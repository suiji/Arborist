// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CART_SFCART_H
#define CART_SFCART_H

/**
   @file sfcart.h

   @brief Manages CART-specific node splitting across the tree frontier.

   @author Mark Seligman

 */

#include "splitcoord.h"
#include "typeparam.h"
#include "sumcount.h"
#include "splitfrontier.h"

#include <vector>

/**
   @brief Splitting facilities specific regression trees.
 */
class SFCartReg : public SplitFrontier {
  // Bridge-supplied monotone constraints.  Length is # numeric predictors
  // or zero, if none so constrained.
  static vector<double> mono;

  // Per-level vector of uniform variates.
  vector<double> ruMono;

  void split(class SplitCand& cand);

 public:

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
  
  SFCartReg(const class SummaryFrame* frame_,
        class Frontier* frontier_,
	const class Sample* sample);

  ~SFCartReg();
  void setRunOffsets(const vector<unsigned int>& safeCount);
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
class SFCartCtg : public SplitFrontier {
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
  void split(class SplitCand& cand);

  /**
     @brief RunSet initialization utitlity.

     @param safeCount gives a conservative per-predictor count of distinct runs.
   */
  void setRunOffsets(const vector<unsigned int>& safeCount);


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

  SFCartCtg(const class SummaryFrame* frame_,
        class Frontier* frontier_,
	const class Sample* sample,
	PredictorT nCtg_);
  ~SFCartCtg();


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
