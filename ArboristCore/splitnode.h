// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_SPLITNODE_H
#define ARBORIST_SPLITNODE_H

/**
   @file splitnode.h

   @brief Maintains per-node splitting parameters and directs splitting
   of nodes across selected predictors.

   @author Mark Seligman

 */

#include "typeparam.h"
#include <vector>

/**
   @brief Per-predictor splitting facilities.
 */
// Currently implemented in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitNode {
  const class RowRank *rowRank;
  void setPrebias(class IndexLevel *index);
  
 protected:
  const class FrameTrain *frameTrain;
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

  SplitNode(const class FrameTrain *_frameTrain,
	    const class RowRank *_rowRank,
	    unsigned int bagCount);

  void scheduleSplits(const class IndexLevel *index,
		      const class Level *levelFront);

  /**
     @brief Emplaces new candidate with specified coordinates.
   */
  void preschedule(unsigned int splitIdx,
		   unsigned int predIdx,
		   unsigned int bufIdx);

  /**
     @brief Pass-through to row-rank method.

     @param cand is the candidate.

     @return rank of dense value, if candidate's predictor has one.
   */
  unsigned int denseRank(const SplitCand* cand) const;

  
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
                unsigned int nSplitNode) const;
  
  virtual void splitCandidates(const class SamplePred *samplePred) = 0;
  virtual ~SplitNode();
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
class SPReg : public SplitNode {
  static unsigned int predMono;
  static vector<double> mono;
  vector<double> ruMono;

  void splitCandidates(const class SamplePred *samplePred);
  int getMonoMode(unsigned int splitIdx,
	       unsigned int predIdx) const;

 public:
  static void Immutables(const vector<double> &feMono);
  static void DeImmutables();
  SPReg(const class FrameTrain *_frameTrain,
	const class RowRank *_rowRank,
	unsigned int bagCount);
  ~SPReg();
  void setRunOffsets(const vector<unsigned int> &safeCount);
  void levelPreset(class IndexLevel *index);
  void levelClear();

  int getMonoMode(const class SplitCand* cand) const;

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
class SPCtg : public SplitNode {
// Numerical tolerances taken from A. Liaw's code:
  static constexpr double minDenom = 1.0e-5;
  static constexpr double minSumL = 1.0e-8;
  static constexpr double minSumR = 1.0e-5;

  const unsigned int nCtg;
  vector<double> sumSquares; // Per-level sum of squares, by split.
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
     @brief Retrieves the type-relative index of a numerical predictor.

     @return placement-adjusted index.
   */
  unsigned int getNumIdx(unsigned int predIdx) const;



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
  vector<double> ctgSum; // Per-level sum, by split/category pair.
  SPCtg(const class FrameTrain *_frameTrain,
	const class RowRank *_rowRank,
	unsigned int bagCount,
	unsigned int _nCtg);
  ~SPCtg();


  /**
     @return number of categories present in training response.
   */
  inline unsigned int getNCtg() const {
    return nCtg;
  }

  
  /**
     @brief Determine whether an ordered pair of sums is acceptably stable
     to appear in the denominator.  Only relevant for instances of extreme
     case weighting.  Currently unused.

     @return true iff both sums suitably stable.
   */
  inline bool stableSum(double sumL, double sumR) const {
    return sumL > minSumL && sumR > minSumR;
  }



  /**
     @brief Determines whether a pair of sums is acceptably stable to appear
     in the denominators.  Only relevant for instances of extreme case
     weighting.  Currently unused.

     @return true iff both sums suitably stable.
   */
  inline bool stableDenom(double sumL, double sumR) const {
    return sumL > minDenom && sumR > minDenom;
  }
  


  /**
     @brief Provides slice into sum vector for a node.

     @param splitIdx is the node index.

     @return raw pointer to per-category sum vector for node.
   */
  double* getSumSlice(const class SplitCand* cand);


  /**
     @brief Provides slice into accumulation vector for a node/predictor
     pair.

     @param splitIdx is the node index.

     @param predIdx is the predictor index.

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
