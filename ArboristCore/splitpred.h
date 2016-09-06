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

#include "param.h"
#include <vector>


/**
   @brief Encapsulates information needed to drive splitting.
 */
class SplitCoord {
  unsigned int splitPos; // Position in containing vector.
  unsigned int levelIdx;
  unsigned int predIdx;
  unsigned int idxStart; // Per node.
  unsigned int sCount;  // Per node.
  double sum; // Per node.
  double preBias; // Per node.
  unsigned int setIdx;  // Per pair.
  unsigned int denseCount; // Per pair:  post restage.
  unsigned int idxEnd; // Per pair:  post restage.
  unsigned char bufIdx; // Per pair.
 public:

  void InitEarly(unsigned int _splitPos, unsigned int _levelIdx, unsigned int _predIdx, unsigned int _bufIdx, unsigned int _setIdx);
  void InitLate(const class Bottom *bottom, const class IndexNode indexNode[]);
  
  void Split(const class SPReg *spReg, const class Bottom *bottom, const class SamplePred *samplePred, const class IndexNode indexNode[]);
  void Split(class SPCtg *spCtg, const class Bottom *bottom, const class SamplePred *samplePred, const class IndexNode indexNode[]);
  void SplitNum(const class SPReg *splitReg, const class Bottom *bottom, const class SPNode spn[]);
  void SplitNum(class SPCtg *splitCtg, const class Bottom *bottom, const class SPNode spn[]);
  bool SplitNum(const class SPReg *spReg, const class SPNode spn[], class SplitNux &nux);
  bool SplitNum(const class SPNode spn[], class SplitNux &nux);
  bool SplitNumMono(bool increasing, const class SPNode spn[], class SplitNux &nux);
  bool SplitNum(class SPCtg *spCtg, const class SPNode spn[], class SplitNux &nux);

  void SplitFac(const class SPReg *splitReg, const class Bottom *bottom, const class SPNode spn[]);
  void SplitFac(const class SPCtg *splitCtg, const class Bottom *bottom, const class SPNode spn[]);
  bool SplitFac(const class SPReg *spReg, const class SPNode spn[], unsigned int &runCount, class SplitNux &nux);
  bool SplitFac(const class SPCtg *spCtg, const class SPNode spn[], unsigned int &runCount, class SplitNux &nux);
  bool SplitBinary(class RunSet *runSet, const class SPCtg *spCtg, class SplitNux &nux);
  bool SplitRuns(class RunSet *runSet, const class SPCtg *spCtg, class SplitNux &nux);

  unsigned int RunsReg(class RunSet *runSet, const class SPNode spn[], unsigned int denseRank) const;
  bool HeapSplit(class RunSet *runSet, class SplitNux &nux) const;
  unsigned int RunsCtg(class RunSet *runSet, const SPNode spn[], unsigned int denseRank) const;
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
  static unsigned int predFixed;
  static const double *predProb;

  void SetPrebias(class IndexNode indexNode[]);
  void SplitFlags(bool unsplitable[]);
  void ScheduleProb(unsigned int levelIdx, const double ruPred[], std::vector<unsigned int> &safeCount);
  void ScheduleFixed(unsigned int levelIdx, const double ruPred[], class BHPair heap[], std::vector<unsigned int> &safeCount);
  bool ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, std::vector<unsigned int> &safeCount);
  
 protected:
  static unsigned int nPred;
  const unsigned int bagCount;
  class Bottom *bottom;
  unsigned int levelCount; // # subtree nodes at current level.
  class Run *run;
  std::vector<SplitCoord> splitCoord; // Schedule of splits.
  void Splitable(const bool unsplitable[], std::vector<unsigned int> &safeCount);
 public:
  class SamplePred *samplePred;
  SplitPred(const class RowRank *_rowRank, class SamplePred *_samplePred, unsigned int bagCount);
  static void Immutables(unsigned int _nPred, unsigned int _ctgWidth, unsigned int _predFixed, const double _predProb[], const double _regMono[]);
  static void DeImmutables();
  unsigned int DenseRank(unsigned int predIdx) const;

  class Run *Runs() {
    return run;
  }

  class RunSet *RSet(unsigned int setIdx) const;

  void SetBottom(class Bottom *_bottom) {
    bottom = _bottom;
  }

  virtual void Split(const class IndexNode indexNode[]) = 0;
  
  virtual ~SplitPred();
  virtual void LevelInit(class Index *index, class IndexNode indexNode[], unsigned int levelCount);
  virtual void RunOffsets(const std::vector<unsigned int> &safeCounts) = 0;
  virtual bool *LevelPreset(const class Index *index) = 0;
  virtual double Prebias(unsigned int levelIdx, unsigned int sCount, double sum) = 0;
  virtual void LevelClear();
};


/**
   @brief Splitting facilities specific regression trees.
 */
class SPReg : public SplitPred {
  static unsigned int predMono;
  static const double *feMono;
  double *ruMono;

  void Split(const class IndexNode indexNode[]);

 public:
  static void Immutables(unsigned int _nPred, const double *_mono);
  static void DeImmutables();
  SPReg(const class RowRank *_rowRank, class SamplePred *_samplePred, unsigned int bagCount);
  ~SPReg();
  int MonoMode(unsigned int splitIdx, unsigned int predIdx) const;
  void RunOffsets(const std::vector<unsigned int> &safeCount);
  bool *LevelPreset(const class Index *index);
  double Prebias(unsigned int spiltIdx, unsigned int sCount, double sum);
  void LevelInit(class Index *index, class IndexNode indexNode[], unsigned int levelCount);
  void LevelClear();
};


/**
   @brief Splitting facilities for categorical trees.
 */
class SPCtg : public SplitPred {
  static unsigned int ctgWidth;
  double *ctgSum; // Per-level sum, by split/category pair.
  double *ctgSumR; // Numeric predictors:  sum to right.
  double *sumSquares; // Per-level sum of squares, by split.
  const std::vector<class SampleNode> &sampleCtg;
  bool *LevelPreset(const class Index *index);
  double Prebias(unsigned int levelIdx, unsigned int sCount, double sum);
  void LevelClear();
  void Split(const class IndexNode indexNode[]);
  void RunOffsets(const std::vector<unsigned int> &safeCount);
  void SumsAndSquares(const class Index *index, bool unsplitable[]);
  unsigned int LHBits(unsigned int lhBits, unsigned int pairOffset, unsigned int depth, unsigned int &lhSampCt);
  void LevelInitSumR();


 public:
// Numerical tolerances taken from A. Liaw's code:
  static constexpr double minDenom = 1.0e-5;
  static constexpr double minSumL = 1.0e-8;
  static constexpr double minSumR = 1.0e-5;

  SPCtg(const class RowRank *_rowRank, class SamplePred *_samplePred, const std::vector<class SampleNode> &_sampleCtg, unsigned int bagCount);
  ~SPCtg();
  static void Immutables(unsigned int _ctgWidth);
  static void DeImmutables();

 public:
  /**
     @brief Looks up node values by category.

     @param levelIdx is the level-relative node index.

     @param ctg is the category.

     @return Sum of index node values at level index, category.
   */
  inline double CtgSum(unsigned int levelIdx, unsigned int ctg) const {
    return ctgSum[levelIdx * ctgWidth + ctg];
  }


  /**
     @brief Records sum of proxy values at 'yCtg' strictly to the right and updates the
     subaccumulator by the current proxy value.

     @param numIdx is contiguouly-numbered numerical index of the predictor.

     @param levelIdx is the level-relative node index.

     @param yCtg is the categorical response value.

     @param ySum is the proxy response value.

     @return recorded sum.
  */
  inline double CtgSumRight(unsigned int levelIdx, unsigned int numIdx, unsigned int yCtg, double ySum) {
    int off = numIdx * levelCount * ctgWidth + levelIdx * ctgWidth + yCtg;
    double val = ctgSumR[off];
    ctgSumR[off] = val + ySum;

    return val;
  }

  static inline unsigned int CtgWidth() {
    return ctgWidth;
  }

  
  double SumSquares(unsigned int levelIdx) {
    return sumSquares[levelIdx];
  }
};


#endif
