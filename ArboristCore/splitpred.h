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
   @brief Encapsulates information needed to drive splitting.
 */
class SplitCoord {
  unsigned int vecPos; // Position in containing vector.
  unsigned int splitIdx;
  unsigned int predIdx;
  unsigned int idxStart; // Per node.
  unsigned int sCount;  // Per node.
  double sum; // Per node.
  double preBias; // Per node.
  unsigned int setIdx;  // Per pair.
  unsigned int implicit;  // Per pair:  post restage.
  unsigned int idxEnd; // Per pair:  post restage.
  unsigned char bufIdx; // Per pair.
 public:

  void InitEarly(unsigned int splitIdx,
		 unsigned int predIdx,
		 unsigned int bufIdx);
  void Schedule(const class Level *levelFront,
		const class IndexLevel *indexLevel,
		unsigned int noSet,
		vector<unsigned int> &runCount,
		vector<SplitCoord> &sc2);
  void InitLate(const class Level *levelFront,
		const class IndexLevel *index,
		unsigned int _splitPos,
		unsigned int _setIdx);

  void Split(const class SPReg *spReg,
	     const class SamplePred *samplePred);
  void Split(class SPCtg *spCtg,
	     const class SamplePred *samplePred);
  void SplitNum(const class SPReg *splitReg,
		const class SampleRank spn[]);
  void SplitNum(class SPCtg *splitCtg,
		const class SampleRank spn[]);
  bool SplitNum(const class SPReg *spReg,
		const class SampleRank spn[],
		class NuxLH &nux);
  bool SplitNum(const class SampleRank spn[],
		class NuxLH &nux);
  bool SplitNumDense(const class SampleRank spn[],
		     const class SPReg *spReg,
		     class NuxLH &nux);
  bool SplitNumDenseMono(bool increasing,
			 const class SampleRank spn[],
			 const class SPReg *spReg,
			 class NuxLH &nux);
  bool SplitNumMono(bool increasing,
		    const class SampleRank spn[],
		    class NuxLH &nux);
  bool SplitNum(class SPCtg *spCtg,
		const class SampleRank spn[],
		class NuxLH &nux);
  bool NumCtgDense(class SPCtg *spCtg,
		   const class SampleRank spn[],
		   class NuxLH &nux);
  bool NumCtg(class SPCtg *spCtg,
	      const class SampleRank spn[],
	      class NuxLH &nux);
  unsigned int NumCtgGini(SPCtg *spCtg,
			  const class SampleRank spn[],
			  unsigned int idxNext,
			  unsigned int idxFinal,
			  unsigned int &sCountL,
			  unsigned int &rkRight,
			  double &sumL,
			  double &ssL,
			  double &ssR,
			  double &maxGini,
			  unsigned int &rankLH,
			  unsigned int &rankRH,
			  unsigned int &rhInf);
  void SplitFac(const class SPReg *splitReg,
		const class SampleRank spn[]);
  void SplitFac(const class SPCtg *splitCtg,
		const class SampleRank spn[]);
  bool SplitFac(const class SPReg *spReg,
		const class SampleRank spn[],
		class NuxLH &nux);
  bool SplitFac(const class SPCtg *spCtg,
		const class SampleRank spn[],
		class NuxLH &nux);
  bool SplitBinary(const class SPCtg *spCtg,
		   class RunSet *runSet,
		   class NuxLH &nux);
  bool SplitRuns(const class SPCtg *spCtg,
		 class RunSet *runSet,
		 class NuxLH &nux);

  void RunsReg(class RunSet *runSet,
	       const class SampleRank spn[],
	       unsigned int denseRank) const;
  bool HeapSplit(class RunSet *runSet,
		 class NuxLH &nux) const;
  void RunsCtg(const class SPCtg *spCtg,
	       class RunSet *runSet,
	       const SampleRank spn[]) const;
};


/**
   @brief Per-predictor splitting facilities.
 */
// Predictor-specific implementation of node.
// Currently available in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitPred {
  const class RowRank *rowRank;
  
 protected:
  const class FrameTrain *frameTrain;
  const unsigned int bagCount;
  const unsigned int noSet; // Unreachable setIdx for SplitCoord.
  unsigned int splitCount; // # subtree nodes at current level.
  class Run *run;
  vector<SplitCoord> splitCoord; // Schedule of splits.
  void ArgMax(vector<class SSNode> &argMax);

public:
  class SplitSig *splitSig;

  SplitPred(const class FrameTrain *_frameTrain,
	    const class RowRank *_rowRank,
	    unsigned int bagCount);

  void ScheduleSplits(const class IndexLevel *index,
		      const class Level *levelFront);
  void Preschedule(unsigned int splitIdx,
		   unsigned int predIdx,
		   unsigned int bufIdx);
  unsigned int DenseRank(unsigned int predIdx) const;
  bool IsFactor(unsigned int predIdx) const;
  void SSWrite(unsigned int levelIdx,
	       unsigned int predIdx,
	       unsigned int setPos,
	       unsigned int bufIdx,
	       const class NuxLH &nux) const;

  class Run *Runs() {
    return run;
  }

  class RunSet *RSet(unsigned int setIdx) const;
  void LevelInit(class IndexLevel *index);
  void Split(const class SamplePred *samplePred,
	     vector<class SSNode> &argMax);

  virtual void Split(const class SamplePred *samplePred) = 0;
  virtual ~SplitPred();
  virtual void RunOffsets(const vector<unsigned int> &safeCounts) = 0;
  virtual void LevelPreset(class IndexLevel *index) = 0;
  virtual double Prebias(unsigned int splitIdx, double sum, unsigned int sCount) const = 0;
  virtual void LevelClear();
};


/**
   @brief Splitting facilities specific regression trees.
 */
class SPReg : public SplitPred {
  static unsigned int predMono;
  static vector<double> mono;
  double *ruMono;

  void Split(const class SamplePred *samplePred);

 public:
  unsigned int Residuals(const SampleRank spn[],
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
  void RunOffsets(const vector<unsigned int> &safeCount);
  void LevelPreset(class IndexLevel *index);
  void LevelClear();

  
  /**
     @brief Weighted-variance pre-bias computation for regression response.

     @param sum is the sum of samples subsumed by the index node.

     @param sCount is the number of samples subsumed by the index node.

     @return sum squared, divided by sample count.
  */
  inline double Prebias(unsigned int splitIdx,
			double sum,
			unsigned int sCount) const {
    return (sum * sum) / sCount;
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
  void LevelPreset(class IndexLevel *index);
  void LevelClear();
  void Split(const class SamplePred *samplePred);
  void RunOffsets(const vector<unsigned int> &safeCount);
  unsigned int LHBits(unsigned int lhBits,
		      unsigned int pairOffset,
		      unsigned int depth,
		      unsigned int &lhSampCt);

  void LevelInitSumR(unsigned int nPredNum);


  /**
     @brief Gini pre-bias computation for categorical response.

     @param splitIdx is the level-relative node index.

     @param sum is the sum of samples subsumed by the index node.

     @param sCount is the number of samples subsumed by the index node.

     @return sum of squares divided by sum.
  */
  inline double Prebias(unsigned int splitIdx,
			double sum,
			unsigned int sCount) const {
    return sumSquares[splitIdx] / sum;
  }


 public:
  SPCtg(const class FrameTrain *_frameTrain,
	const class RowRank *_rowRank,
	unsigned int bagCount,
	unsigned int _nCtg);
  ~SPCtg();
  unsigned int Residuals(const SampleRank spn[],
			 unsigned int levelIdx,
			 unsigned int idxStart,
			 unsigned int idxEnd,
			 unsigned int denseRank,
			 bool &denseLeft,
			 bool &denseRight,
			 double &sumDense,
			 unsigned int &sCountDense,
			 vector<double> &ctgSumDense) const;

  void ApplyResiduals(unsigned int levelIdx,
		      unsigned int predIdx,
		      double &ssL,
		      double &ssr,
		      vector<double> &sumDenseCtg);

  
  unsigned int NumIdx(unsigned int predIdx) const;

  /**
     @return number of categories present in training response.
   */
  inline unsigned int CtgWidth() const {
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
  inline double CtgSum(unsigned int levelIdx, unsigned int ctg) const {
    return ctgSum[levelIdx * nCtg + ctg];
  }


  /**
     @return column of category sums at index.
   */
  inline const double *ColumnSums(unsigned int levelIdx) const {
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
  inline double CtgSumAccum(unsigned int levelIdx,
			    unsigned int numIdx,
			    unsigned int yCtg,
			    double ySum) {
    int off = numIdx * splitCount * nCtg + levelIdx * nCtg + yCtg;
    double val = ctgSumAccum[off];
    ctgSumAccum[off] = val + ySum;

    return val;
  }


  double SumSquares(unsigned int levelIdx) const {
    return sumSquares[levelIdx];
  }
};


#endif
