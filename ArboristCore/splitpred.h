// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_SPLITPRED_H
#define ARBORIST_SPLITPRED_H

/**
   @file splitpred.h

   @brief Class definitions for the four types of predictor splitting:  {regression, categorical} x {numerical, factor}.

   @author Mark Seligman

 */

#include "typeparam.h"
#include <vector>


/**
   @brief Per-predictor splitting facilities.
 */
// Predictor-specific implementation of node.
// Currently available in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitPred {
  const class RowRank *rowRank;
  void setPrebias(class IndexLevel *index);
  
 protected:
  const class FrameTrain *frameTrain;
  const unsigned int bagCount;
  const unsigned int noSet; // Unreachable setIdx for SplitCand.
  unsigned int splitCount; // # subtree nodes at current level.
  unique_ptr<class Run> run;
  vector<class SplitCand> splitCand; // Schedule of splits.

  vector<double> prebias; // Initial information threshold.
  // Per-split accessors for candidate vector.  Set to splitCount
  // and cleared after use:
  vector<unsigned int> candOff;  // Lead candidate position.
  vector<unsigned int> nCand;  // Number of candidates.

public:
  //  unique_ptr<class SplitSig> splitSig;

  SplitPred(const class FrameTrain *_frameTrain,
	    const class RowRank *_rowRank,
	    unsigned int bagCount);

  void scheduleSplits(const class IndexLevel *index,
		      const class Level *levelFront);

  /**
     @brief Emplaces new candidate with specified coordinates.
   */
  void preSchedule(unsigned int splitIdx,
		   unsigned int predIdx,
		   unsigned int bufIdx);

  /**
     @brief Pass-through to row-rank method.

     @param predIdx is a predictor index.

     @return rank of dense value, if predictor has one.
   */
  unsigned int denseRank(unsigned int predIdx) const;

  /**
     @brief Pass-through to frame-map method.

     @param predIdx is a predictor index.

     @return true iff predictor is a factor.
   */
  bool isFactor(unsigned int predIdx) const;

  class Run *getRuns() {
    return run.get();
  }

  /**
     @brief Accessor for noSet member.
   */
  inline unsigned int getNoSet() const {
    return noSet;
  }
  
  inline double getPrebias(unsigned int splitIdx) const {
    return prebias[splitIdx];
  }

  class RunSet *rSet(unsigned int setIdx) const;
  void levelInit(class IndexLevel *index);
  vector<class SplitCand> split(const class SamplePred *samplePred);


  vector<class SplitCand> maxCandidates();
  
  void maxSplit(SplitCand &candMax,
                unsigned int splitOff,
                unsigned int nSplitPred) const;
  
  virtual void splitCandidates(const class SamplePred *samplePred) = 0;
  virtual ~SplitPred();
  virtual void setRunOffsets(const vector<unsigned int> &safeCounts) = 0;
  virtual void levelPreset(class IndexLevel *index) = 0;

  virtual void setPrebias(unsigned int splitIdx,
                            double sum,
                          unsigned int sCount) = 0;

  virtual void levelClear();
};


/**
   @brief Splitting facilities specific regression trees.
 */
class SPReg : public SplitPred {
  static unsigned int predMono;
  static vector<double> mono;
  vector<double> ruMono;

  void splitCandidates(const class SamplePred *samplePred);

 public:
  unsigned int Residuals(const class SampleRank spn[],
			 unsigned int idxStart,
			 unsigned int idxEnd,
			 unsigned int denseRank,
			 unsigned int &denseLeft,
			 unsigned int &denseRight,
			 double &sumDense,
			 unsigned int &sCountDense) const;

  static void Immutables(const vector<double> &feMono);
  static void DeImmutables();
  SPReg(const class FrameTrain *_frameTrain,
	const class RowRank *_rowRank,
	unsigned int bagCount);
  ~SPReg();
  int MonoMode(unsigned int splitIdx,
	       unsigned int predIdx) const;
  void setRunOffsets(const vector<unsigned int> &safeCount);
  void levelPreset(class IndexLevel *index);
  void levelClear();

  
  /**
     @brief Weighted-variance pre-bias computation for regression response.

     @param sum is the sum of samples subsumed by the index node.

     @param sCount is the number of samples subsumed by the index node.

     @return sum squared, divided by sample count.
  */
  inline void setPrebias(unsigned int splitIdx,
			double sum,
			unsigned int sCount) {
    prebias[splitIdx] = (sum * sum) / sCount;
  }

};


/**
   @brief Splitting facilities for categorical trees.
 */
class SPCtg : public SplitPred {
// Numerical tolerances taken from A. Liaw's code:
  static constexpr double minDenom = 1.0e-5;
  static constexpr double minSumL = 1.0e-8;
  static constexpr double minSumR = 1.0e-5;

  const unsigned int nCtg;
  vector<double> sumSquares; // Per-level sum of squares, by split.
  vector<double> ctgSum; // Per-level sum, by split/category pair.
  vector<double> ctgSumAccum; // Numeric predictors:  accumulate sums.
  void levelPreset(class IndexLevel *index);
  void levelClear();
  void splitCandidates(const class SamplePred *samplePred);
  void setRunOffsets(const vector<unsigned int> &safeCount);
  unsigned int LHBits(unsigned int lhBits,
		      unsigned int pairOffset,
		      unsigned int depth,
		      unsigned int &lhSampCt);

  void levelInitSumR(unsigned int nPredNum);


  /**
     @brief Gini pre-bias computation for categorical response.

     @param splitIdx is the level-relative node index.

     @param sum is the sum of samples subsumed by the index node.

     @param sCount is the number of samples subsumed by the index node.

     @return sum of squares divided by sum.
  */
  inline void setPrebias(unsigned int splitIdx,
			double sum,
			unsigned int sCount) {
    prebias[splitIdx] = sumSquares[splitIdx] / sum;
  }


 public:
  SPCtg(const class FrameTrain *_frameTrain,
	const class RowRank *_rowRank,
	unsigned int bagCount,
	unsigned int _nCtg);
  ~SPCtg();
  unsigned int Residuals(const class SampleRank spn[],
			 unsigned int levelIdx,
			 unsigned int idxStart,
			 unsigned int idxEnd,
			 unsigned int denseRank,
			 bool &denseLeft,
			 bool &denseRight,
			 double &sumDense,
			 unsigned int &sCountDense,
			 vector<double> &ctgSumDense) const;

  void applyResiduals(unsigned int levelIdx,
		      unsigned int predIdx,
		      double &ssL,
		      double &ssr,
		      vector<double> &sumDenseCtg);


  /**
     @brief Retrieves the type-relative index of a numerical predictor.

     @return placement-adjusted index.
   */
  unsigned int getNumIdx(unsigned int predIdx) const;

  /**
     @return number of categories present in training response.
   */
  inline unsigned int getCtgWidth() const {
    return nCtg;
  }

  
  /**
     @brief Determine whether a pair of square-sums is acceptably stable
     for a gain computation.

     @return true iff both sums suitably stable.
   */
  inline bool StableSums(double sumL, double sumR) const {
    return sumL > minSumL && sumR > minSumR;
  }



  /**
     @brief Determines whether a pair of sums is acceptably stable to appear
     in the denominators of a gain computation.

     @return true iff both sums suitably stable.
   */
  inline bool StableDenoms(double sumL, double sumR) const {
    return sumL > minDenom && sumR > minDenom;
  }
  

  /**
     @brief Looks up node values by category.

     @param levelIdx is the level-relative node index.

     @param ctg is the category.

     @return Sum of index node values at level index, category.
   */
  inline double getCtgSum(unsigned int levelIdx, unsigned int ctg) const {
    return ctgSum[levelIdx * nCtg + ctg];
  }


  /**
     @return column of category sums at index.
   */
  inline const double *getColumnSums(unsigned int levelIdx) const {
    return &ctgSum[levelIdx * nCtg];
  }

  
  /**
     @brief Accumulates sum of proxy values at 'yCtg' walking strictly
     in a given direction and updates the subaccumulator by the current
     proxy value.

     @param levelIdx is the level-relative node index.

     @param numIdx is contiguouly-numbered numerical index of the predictor.

     @param yCtg is the categorical response value.

     @param ySum is the proxy response value.

     @return current partial sum.
  */
  inline double accumCtgSum(unsigned int levelIdx,
			    unsigned int numIdx,
			    unsigned int yCtg,
			    double ySum) {
    int off = numIdx * splitCount * nCtg + levelIdx * nCtg + yCtg;
    double val = ctgSumAccum[off];
    ctgSumAccum[off] = val + ySum;

    return val;
  }


  double getSumSquares(unsigned int levelIdx) const {
    return sumSquares[levelIdx];
  }
};


#endif
