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
   @brief Per-predictor splitting facilities.
 */
// Predictor-specific implementation of node.
// Currently available in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitPred {
  static unsigned int predFixed;
  static const double *predProb;

  void SetPrebias(class IndexNode indexNode[]);
  void SplitFlags(bool unsplitable[]);
  void SplitPredProb(unsigned int levelIdx, const double ruPred[], std::vector<unsigned int> &safeCount);
  void SplitPredFixed(unsigned int levelIdx, const double ruPred[], class BHPair heap[], std::vector<unsigned int> &safeCount);

 protected:
  static unsigned int nPred;
  class Bottom *bottom;
  unsigned int levelCount; // # subtree nodes at current level.
  class Run *run;
  void Splitable(const bool unsplitable[], std::vector<unsigned int> &safeCount);
 public:
  class SamplePred *samplePred;
  SplitPred(class SamplePred *_samplePred, unsigned int bagCount);
  static void Immutables(unsigned int _nPred, unsigned int _ctgWidth, unsigned int _predFixed, const double _predProb[], const double _regMono[]);
  static void DeImmutables();

  class Run *Runs() {
    return run;
  }
  void SetBottom(class Bottom *_bottom) {
    bottom = _bottom;
  }

  void Split(unsigned int splitIdx, const class IndexNode indexNode[], const class SPNode spn[]);
  
  virtual ~SplitPred();
  virtual void LevelInit(class Index *index, class IndexNode indexNode[], unsigned int levelCount);
  virtual void RunOffsets(const std::vector<unsigned int> &safeCounts) = 0;
  virtual bool *LevelPreset(const class Index *index) = 0;
  virtual double Prebias(unsigned int levelIdx, unsigned int sCount, double sum) = 0;
  virtual void LevelClear();

  virtual void SplitNum(unsigned int splitIdx, const class IndexNode indexNode[], const class SPNode spn[]) = 0;
  virtual void SplitFac(unsigned int splitIdx, const class IndexNode indexNode[], const class SPNode spn[]) = 0;
};


/**
   @brief Splitting facilities specific regression trees.
 */
class SPReg : public SplitPred {
  static unsigned int predMono;
  static const double *feMono;
  double *ruMono;

  int MonoMode(unsigned int splitIdx);
  void SplitHeap(const class IndexNode *indexNode, const class SPNode spn[], unsigned int predIdx);
  void Split(const class IndexNode indexNode[], class SPNode *nodeBase);
  void SplitNum(unsigned int splitIdx, const class IndexNode indexNode[], const class SPNode spn[]);
  void SplitNumWV(unsigned int splitIdx, const class IndexNode *indexNode, const class SPNode spn[]);
  void SplitNumMono(unsigned int splitIdx, const class IndexNode *indexNode, const class SPNode spn[], bool increasing);
  void SplitFac(unsigned int splitIdx, const class IndexNode indexNode[], const class SPNode *nodeBase);
  void SplitFacWV(unsigned int splitIdx, const class IndexNode *indexNode, const class SPNode spn[]);
  unsigned int BuildRuns(class RunSet *runSet, const class SPNode spn[], unsigned int start, unsigned int end);
  unsigned int HeapSplit(class RunSet *runSet, double sum, unsigned int sCountNode, unsigned int &lhIdxCount, double &maxGini);


 public:
  static void Immutables(unsigned int _nPred, const double *_mono);
  static void DeImmutables();
  SPReg(class SamplePred *_samplePred, unsigned int bagCount);
  ~SPReg();
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
// Numerical tolerances taken from A. Liaw's code:
  static constexpr double minDenom = 1.0e-5;
  static constexpr double minSumL = 1.0e-8;
  static constexpr double minSumR = 1.0e-5;
  const class SampleNode *sampleCtg;
  bool *LevelPreset(const class Index *index);
  double Prebias(unsigned int levelIdx, unsigned int sCount, double sum);
  void LevelClear();
  void Split(const class IndexNode indexNode[], class SPNode *nodeBase);
  void RunOffsets(const std::vector<unsigned int> &safeCount);
  void SumsAndSquares(const class Index *index, bool unsplitable[]);
  unsigned int LHBits(unsigned int lhBits, unsigned int pairOffset, unsigned int depth, unsigned int &lhSampCt);

  /**
     @brief Looks up node values by category.

     @param levelIdx is the level-relative node index.

     @param ctg is the category.

     @return Sum of index node values at level index, category.
   */
  inline double CtgSum(unsigned int levelIdx, unsigned int ctg) {
    return ctgSum[levelIdx * ctgWidth + ctg];
  }

  void LevelInitSumR();
  void SplitNum(unsigned int splitIdx, const class IndexNode indexNode[], const class SPNode spn[]);
  void SplitNumGini(unsigned int splitIdx, const class IndexNode *indexNode, const class SPNode spn[]);
  unsigned int SplitBinary(class RunSet *runSet, unsigned int levelIdx, double sum, double &maxGini, unsigned int &sCount);
  unsigned int BuildRuns(class RunSet *runSet, const class SPNode spn[], unsigned int start, unsigned int end);
  unsigned int SplitRuns(class RunSet *runSet, unsigned int levelIdx, double sum, double &maxGini, unsigned int &lhSampCt);
  
 public:
  SPCtg(class SamplePred *_samplePred, class SampleNode _sampleCtg[], unsigned int bagCount);
  ~SPCtg();
  static void Immutables(unsigned int _ctgWidth);
  static void DeImmutables();
  
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

 public:
  static inline unsigned int CtgWidth() {
    return ctgWidth;
  }
  void SplitFac(unsigned int splitIdx, const class IndexNode indexNode[], const class SPNode spn[]);
  void SplitFacGini(unsigned int splitIdx, const class IndexNode *indexNode, const class SPNode spn[]);
};


#endif
