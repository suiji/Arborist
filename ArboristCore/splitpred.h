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
   @brief Pair-based splitting information.
 */
class SPPair {
  int splitIdx;
  unsigned int predIdx;
  unsigned int pairIdx;
  int setIdx; // Nonnegative iff nontrivial run.
 public:
  
  inline void Init(int _splitIdx, unsigned int _predIdx, unsigned int _pairIdx) {
    splitIdx = _splitIdx;
    predIdx = _predIdx;
    pairIdx = _pairIdx;
  }

  
  inline int SplitIdx() const {
    return splitIdx;
  }


  inline unsigned int PairIdx() const {
    return pairIdx;
  }


  inline void Coords(int &_splitIdx, unsigned int &_predIdx) const {
    _splitIdx = splitIdx;
    _predIdx = predIdx;
  }

  /**
    @brief Looks up associated pair-run index.

    @return referece to index in PairRun vector.
   */
  inline void SetRSet(int idx) {
    setIdx = idx;
  }

  inline int RSet() const {
    return setIdx;
  }

  void Split(class SplitPred *splitPred, const class IndexNode indexNode[], class SPNode *nodeBase,  const class SamplePred *samplePred, class SplitSig *splitSig);
};


/**
   @brief Per-predictor splitting facilities.
 */
// Predictor-specific implementation of node.
// Currently available in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitPred {
  static unsigned int nPred;
  static int predFixed;
  static std::vector<double> predProb;
  SPPair *spPair;

  void SplitFlags(bool unsplitable[]);
  void SplitPredNull(bool splitFlags[]);
  void SplitPredProb(const double ruPred[], bool splitFlags[]);
  void SplitPredFixed(const double ruPred[], class BHPair heap[], bool splitFlags[]);
  void LevelSplit(const class IndexNode _indexNode[], class SPNode *nodeBase, int splitCount);
  SPPair *PairInit(unsigned int nPred, int &pairCount);
 protected:
  int pairCount;
  int splitCount;
  
  class Run *run;
  bool *splitFlags; // Indexed by pair.

 public:
  const class SamplePred *samplePred;
  SplitPred(class SamplePred *_samplePred);
  static void Immutables(unsigned int _nPred, unsigned int _ctgWidth, int _predFixed, const double _predProb[], const double _regMono[]);
  static void DeImmutables();
  static SplitPred *FactoryReg(class SamplePred *_samplePred);
  static SplitPred *FactoryCtg(class SamplePred *_samplePred, class SampleNode *_sampleCtg);

  void LevelSplit(const class IndexNode indexNode[], unsigned int level, int splitCount, class SplitSig *splitSig);
  void LevelSplit(const class IndexNode indexNode[], class SPNode *nodeBase, int splitCount, class SplitSig *splitSig);  
  void LengthTransmit(int splitIdx, int lNext, int rNext);
  unsigned int &LengthNext(int splitNext, unsigned int predIdx);
  void LengthVec(int splitNext);
  inline void SplitFields(int pairIdx, int &_splitIdx, unsigned int &_predIdx) {
    spPair[pairIdx].Coords(_splitIdx, _predIdx);
  }

  void RLNext(int splitIdx, int lNext, int rNext);
  unsigned int &RLNext(int splitIdx, unsigned int predIdx);
  bool Singleton(int splitIdx, unsigned int predIdx);
  void Split(const class IndexNode indexNode[], class SPNode *nodeBase, class SplitSig *splitSig);

  class Run *Runs() {
    return run;
  }
  
  virtual ~SplitPred();
  virtual void LevelInit(class Index *index, int splitCount);
  virtual void RunOffsets() = 0;
  virtual bool *LevelPreset(const class Index *index) = 0;
  virtual double Prebias(int splitIdx, unsigned int sCount, double sum) = 0;
  virtual void LevelClear();

  virtual void SplitNum(const SPPair *spPair, const class IndexNode indexNode[], const class SPNode spn[], class SplitSig *splitSig) = 0;
  virtual void SplitFac(const SPPair *spPair, const class IndexNode indexNode[], const class SPNode spn[], class SplitSig *splitSig) = 0;
};


/**
   @brief Splitting facilities specific regression trees.
 */
class SPReg : public SplitPred {
  static unsigned int predMono;
  static double *mono;
  double *ruMono;
  ~SPReg();

  int MonoMode(const SPPair *spPair);
  void SplitHeap(const class IndexNode *indexNode, const class SPNode spn[], unsigned int predIdx, class SplitSig *splitSig);
  void Split(const class IndexNode indexNode[], class SPNode *nodeBase, class SplitSig *splitSig);
  void SplitNum(const SPPair *spPair, const class IndexNode indexNode[], const class SPNode spn[], class SplitSig *splitSig);
  void SplitNumWV(const SPPair *spPair, const class IndexNode *indexNode, const class SPNode spn[], class SplitSig *splitSig);
  void SplitNumMono(const SPPair *spPair, const class IndexNode *indexNode, const class SPNode spn[], class SplitSig *splitSig, bool increasing);
  void SplitFac(const SPPair *spPair, const class IndexNode indexNode[], const class SPNode *nodeBase, class SplitSig *splitSig);
  void SplitFacWV(const SPPair *spPair, const class IndexNode *indexNode, const class SPNode spn[], class SplitSig *splitSig);
  unsigned int BuildRuns(class RunSet *runSet, const class SPNode spn[], int start, int end);
  unsigned int HeapSplit(class RunSet *runSet, double sum, unsigned int sCountNode, unsigned int &lhIdxCount, double &maxGini);


 public:
  static void Immutables(unsigned int _nPred, const double *_mono);
  static void DeImmutables();
  SPReg(class SamplePred *_samplePred);
  void RunOffsets();
  bool *LevelPreset(const class Index *index);
  double Prebias(int spiltIdx, unsigned int sCount, double sum);
  void LevelInit(class Index *index, int splitCount);
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
  double Prebias(int splitIdx, unsigned int sCount, double sum);
  void LevelClear();
  void Split(const class IndexNode indexNode[], class SPNode *nodeBase, class SplitSig *splitSig);
  void RunOffsets();
  void SumsAndSquares(const class Index *index, bool unsplitable[]);
  unsigned int LHBits(unsigned int lhBits, int pairOffset, unsigned int depth, int &lhSampCt);

  /**
     @brief Looks up node values by category.

     @param splitIdx is the split index.

     @param ctg is the category.

     @return Sum of index node values at split index, category.
   */
  inline double CtgSum(int splitIdx, unsigned int ctg) {
    return ctgSum[splitIdx * ctgWidth + ctg];
  }

  void LevelInitSumR();
  void SplitNum(const SPPair *spPair, const class IndexNode indexNode[], const class SPNode spn[], class SplitSig *splitSig);
  void SplitNumGini(const SPPair *spPair, const class IndexNode *indexNode, const class SPNode spn[], class SplitSig *splitSig);
  unsigned int SplitBinary(class RunSet *runSet, int splitIdx, double sum, double &maxGini, unsigned int &sCount);
  unsigned int BuildRuns(class RunSet *runSet, const class SPNode spn[], int start, int end);
  unsigned int SplitRuns(class RunSet *runSet, int splitIdx, double sum, double &maxGini, unsigned int &lhSampCt);
  
 public:
  SPCtg(class SamplePred *_samplePred, class SampleNode _sampleCtg[]);
  ~SPCtg();
  static void Immutables(unsigned int _ctgWidth);
  static void DeImmutables();
  
  /**
     @brief Records sum of proxy values at 'yCtg' strictly to the right and updates the
     subaccumulator by the current proxy value.

     @param numIdx is contiguouly-numbered numerical index of the predictor.

     @param splitIdx is the split index.

     @param yCtg is the categorical response value.

     @param ySum is the proxy response value.

     @return recorded sum.
  */
  inline double CtgSumRight(int splitIdx, int numIdx, unsigned int yCtg, double ySum) {
    int off = numIdx * splitCount * ctgWidth + splitIdx * ctgWidth + yCtg;
    double val = ctgSumR[off];
    ctgSumR[off] = val + ySum;

    return val;
  }

 public:
  static inline unsigned int CtgWidth() {
    return ctgWidth;
  }
  void SplitFac(const SPPair *spPair, const class IndexNode indexNode[], const class SPNode spn[], class SplitSig *splitSig);
  void SplitFacGini(const SPPair *spPair, const class IndexNode *indexNode, const class SPNode spn[], class SplitSig *splitSig);
};


#endif
