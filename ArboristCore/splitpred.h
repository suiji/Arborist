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

/**
   @brief Per-predictor splitting facilities.
 */
// Predictor-specific implementation of node.
// Currently available in four flavours depending on response type of node and data
// type of predictor:  { regression, categorical } x { numeric, factor }.
//
class SplitPred {
  void LevelSplit(const class IndexNode _indexNode[], class SPNode *nodeBase, int splitCount);
 protected:
  static int nPred;
  static int nPredNum;
  static int nPredFac;
  static int nFacTot;
  static int predNumFirst;
  static int predNumSup;
  static int predFacFirst;
  static int predFacSup;
  
  bool *splitFlags;
  bool *runFlags;
  static void Immutables();
  void ProbSplitable(int splitCount);
  static int HeapRuns(class FacRunHeap *frHeap, const class SPNode spn[], int splitIdx, int predIdx, int start, int end);
  static int HeapSplit(class FacRunHeap *frHeap, int pairOffset, int depth, double sum, int &sCount, double &maxGini);
 public:
  const class SamplePred *samplePred;
  SplitPred(class SamplePred *_samplePred);
  static SplitPred *FactoryReg(class SamplePred *_samplePred);
  static SplitPred *FactoryCtg(class SamplePred *_samplePred, class SampleNodeCtg *_sampleCtg);
  static void ImmutablesCtg(unsigned int _nRow, int _nSamp, unsigned int _ctgWidth);
  static void ImmutablesReg(unsigned int _nRow, int _nSamp);
  static void DeImmutables();

  void LevelInit(class Index *index, int splitCount);
  void LevelSplit(const class IndexNode indexNode[], int level, int splitCount, class SplitSig *splitSig);
  void LevelSplit(const class IndexNode indexNode[], class SPNode *nodeBase, int splitCount, class SplitSig *splitSig);
  bool *RunFlagReplace(int splitCount);
  void TransmitRun(int splitCount, int predIdx, int splitL, int splitR);
  virtual ~SplitPred();

  virtual void LevelPreset(const class Index *index, int splitCount) = 0;
  virtual double Prebias(int splitIdx, int sCount, double sum) = 0;
  virtual int RunBounds(int splitIdx, int predIdx, int slot, int &start, int &end) = 0;
  virtual void LevelClear() = 0;
  virtual void Split(const class IndexNode indexNode[], class SPNode *nodeBase, int splitCount, class SplitSig *splitSig) = 0;

  
  /**
     @brief Returns address of run value.
   */
  inline bool &RunFlag(int splitCount, int splitIdx, int predIdx, bool _runFlags[]) {
    return _runFlags[splitCount * predIdx + splitIdx];
  }

  
  /**
     @brief Sets the specified run bit in 'runFlags'.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.

     @return void.
  */
  void inline SetPredRun(int splitCount, int splitIdx, int predIdx) {
    RunFlag(splitCount, splitIdx, predIdx, runFlags) = true;
  }

  
  /**
     @brief Reads specified run bit in specified bit vector.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.

     @param _runFlags[] is the run bit vector.

     @return whether specified run bit is set.
   */
  bool inline PredRun(int splitCount, int splitIdx, int predIdx, bool _runFlags[]) {
     return RunFlag(splitCount, splitIdx, predIdx, _runFlags);
  }

  
  
  /**
     @brief Determines whether this split/pred pair is splitable.

     @param splitIdx is the split index.

     @param predIdx is the predictor index.

     @return true iff the  pair is neither in the pred-prob rejection region nor a run.
  */
  bool inline Splitable(int splitCount, int splitIdx, int predIdx) {
    return splitFlags[splitCount * predIdx + splitIdx] && !PredRun(splitCount, splitIdx, predIdx, runFlags);
  }

};


/**
   @brief Splitting facilities specific regression trees.
 */
class SPReg : public SplitPred {
  ~SPReg();
  class FacRunHeap *frHeap;
  void SplitHeap(const class IndexNode *indexNode, const class SPNode spn[], int predIdx, class SplitSig *splitSig);
  void Split(const class IndexNode indexNode[], class SPNode *nodeBase, int splitCount, class SplitSig *splitSig);
  void SplitNum(const class IndexNode indexNode[], class SPNode *nodeBase, int splitCount, int predIdx, class SplitSig *splitSig);
  void SplitNumGini(const class IndexNode *indexNode, const class SPNode spn[], int predIdx, class SplitSig *splitSig);
  void SplitFac(const class IndexNode indexNode[], class SPNode *nodeBase, int splitCount, int predIdx, class SplitSig *splitSig);

 public:
  SPReg(class SamplePred *_samplePred);
  static void Immutables(unsigned int _nRow, int _nSamp);
  static void DeImmutables();
  void LevelPreset(const class Index *index, int splitCount);
  double Prebias(int spiltIdx, int sCount, double sum);
  void LevelClear();
  int RunBounds(int splitIdx, int predIdx, int slot, int &start, int &end);
};


/**
   @brief Splitting facilities for categorical trees.
 */
class SPCtg : public SplitPred {
  const class SampleNodeCtg *sampleCtg;
  class FacRunOrd *frOrd;
  void LevelPreset(const class Index *index, int splitCount);
  double Prebias(int splitIdx, int sCount, double sum);
  void LevelClear();
  void Split(const class IndexNode indexNode[], class SPNode *nodeBase, int splitCount, class SplitSig *splitSig);
 protected:
  static double minDenomNum; // Suggested by Andy Liaw's implementation.
  double *ctgSum; // Per-level sum, by split/category pair.
  double *ctgSumR; // Numeric predictors:  sum to right.
  double *sumSquares; // Per-level sum of squares, by split.
  void SumsAndSquares(const class Index *index, int splitCount);
  int LHBits(unsigned int lhBits, int pairOffset, unsigned int depth, int &lhSampCt);


  /**
     @brief Looks up node values by category.

     @param splitIdx is the split index.

     @param ctg is the category.

     @return Sum of index node values at split index, category.
   */
  inline double CtgSum(int splitIdx, unsigned int ctg) {
    return ctgSum[splitIdx * ctgWidth + ctg];
  }
  void LevelInitSumR(int splitCount);
  void SplitNum(const class IndexNode indexNode[], class SPNode *nodeBase, int splitCount, int predIdx, class SplitSig *splitSig);
  void SplitNumGini(const class IndexNode *indexNode, const class SPNode spn[], int splitCount, int predIdx, class SplitSig *splitSig);

  unsigned int BuildRuns(const class SPNode spn[], int splitIdx, int predIdx, int start, int end);
  unsigned int SplitRuns(int splitIdx, int predIdx, double sum, unsigned int depth, double &maxGini);
  
 public:
  static unsigned int ctgWidth; // Response cardinality:  immutable.
  static void Immutables(unsigned int _nRow, int _nSamp, unsigned int _ctgWidth);
  static void DeImmutables();
  SPCtg(class SamplePred *_samplePred, class SampleNodeCtg _sampleCtg[]);
  ~SPCtg();

  /**
     @brief Records sum of proxy values at 'yCtg' strictly to the right and updates the
     subaccumulator by the current proxy value.

     @param predIdx is the predictor index.  Assumes numerical predictors contiguous.  

     @param splitIdx is the split index.

     @param yCtg is the categorical response value.

     @param yVal is the proxy response value.

     @return recorded sum.
  */
  inline double CtgSumRight(int splitCount, int splitIdx, int predIdx, unsigned int yCtg, double yVal) {
    int off = (predIdx - predNumFirst) * splitCount * ctgWidth + splitIdx * ctgWidth + yCtg;
    double val = ctgSumR[off];
    ctgSumR[off] = val + yVal;

    return val;
  }

 public:
  void SplitFac(const class IndexNode indexNode[], class SPNode *nodeBase, int splitCount, int predIdx, class SplitSig *splitSig);
  void SplitFacGini(const class IndexNode *indexNode, const class SPNode spn[], int predIdx, class SplitSig *splitSig);
  int RunBounds(int splitIdx, int predIdx, int slot, int &start, int &end);
};


#endif
