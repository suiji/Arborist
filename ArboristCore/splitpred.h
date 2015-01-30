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
  static void ProbSplitable(int splitCount);
 protected:
  static SplitPred *splitPred; // Singleton virtual guide.
  static int nPred;
  static int nPredNum;
  static int nPredFac;
  static int nFacTot;
  static int levelMax;

 public:
  static bool *splitFlags;
  static bool *runFlags;
  static void Factory(int _levelMax);
  static void ReFactory(int _levelMax);
  static void DeFactory();
  static void TreeInit();
  static void CheckStorage();
  static void Level(int splitCount, int level);
  static bool PredRun(int splitIdx, int predIdx);
  static void setPredRun(int splitNext, int predIdx);
  static void FactoryReg(int maxWidth);
  static void ReFactoryReg(int _levelMax);
  static int FactoryCtg(int maxWidth, int _ctgWidth);
  static void ReFactoryCtg(int _levelMax);
  static void DeFactoryReg();
  static void DeFactoryCtg();

  static bool PredRun(int splitIdx, int predIdx, int level);
  static void TransmitRun(int splitIdx, int predIdx, int splitL, int splitR, int level);
  static void SetPredRun(int splitIdx, int predIdx, int level, bool val);
  static bool Splitable(int splitIdx, int predIdx, int splitCount, int level);

  virtual ~SplitPred() {}
  virtual void LevelZero() = 0;
  virtual void LevelReset(int splitCount) = 0;
  virtual void RestageAndSplit(int splitCount, int level) = 0;
};

/**
   @brief Splitting facilities specific regression trees.
 */
class SPReg : public SplitPred {
  ~SPReg() {};
 public:
  static void ReFactory(int _levelMax);
  static void Factory(int _levelMax);
  static void DeFactory();
  void LevelZero();
  void LevelReset(int splitCount);
  void RestageAndSplit(int splitCount, int level);
};

/**
   @brief Gini splitting for numerical predictors.
 */
class SPRegNum : public SPReg {
 public:
  static void SplitGini(int predIdx, int splitCount, int level);
  static void Factory();
  static void ReFactory();
  static void DeFactory();
};

/**
   @brief Gini splitting for factor-valued predictors.
 */
class SPRegFac : public SPReg {
 public:
  static void BuildRuns(const class SamplePred samplePred[], int splitIdx, int predIdx, int start, int end);
  static int SplitRuns(int splitIdx, int predIdx, double sum, int &sCount, int &lhIdxCount, double &maxGini);
  static void SplitGini(int predIdx, int splitCount, int level);
  static void Factory();
  static void ReFactory();
  static void DeFactory();
};

/**
   @brief Splitting facilities for categorical trees.
 */
class SPCtg : public SplitPred {
  ~SPCtg() {}
  void LevelZero();
  void LevelReset(int splitCunt);
  void RestageAndSplit(int splitCount, int level);
 protected:
  static int ctgWidth;
 public:
  static void Factory(int _levelMax, int _ctgWidth);
  static void DeFactory();
  static void ReFactory(int _levelMax);
};

/**
   @brief Gini splitting for numerical predictors.
 */
class SPCtgNum : public SPCtg {
 public:
  // Gini coefficient is non-negative:  quotient of non-negative quantities.
  //  static const double giniMin = -1.0e25;

  // Mininum denominator value at which to test a split
  static double minDenom;

  //
  // Numerators do not use updated sumR/sumL values, so update is delayed until
  // current value is recorded.
  //
  static double *ctgSumR;
  /**
     @brief Records sum of proxy values at 'yCtg' strictly to the right and updates the
     subaccumulator by the current proxy value.

     @param predIdx is the predictor index.

     @param splitIdx is the split index.

     @param yCtg is the categorical response value.

     @param yVal is the proxy response value.

     @return recorded sum.
  */
  // TODO:  Reverse first two parameters to conform with similar invocations.
  static inline double CtgSumRight(int predIdx, int splitIdx, int yCtg, double yVal) {
    int off = predIdx * levelMax * ctgWidth + splitIdx * ctgWidth + yCtg;
    double val = ctgSumR[off];
    ctgSumR[off] = val + yVal;

    return val;
  }
 public:
  static void Factory();
  static void ReFactory();
  static void DeFactory();
  static void LevelResetSumR(int splitCount);
  static void SplitGini(int predIdx, int splitCount, int level);
};

/**
   @brief Gini splitting for factor-valued predictors.
*/
class SPCtgFac : public SPCtg {
  static int BuildRuns(const class SamplePred samplePred[], int splitIdx, int predIdx, int start, int end);
  static int SplitRuns(int splitIdx, int predIdx, int splitCount, double sum, int &top, double &maxGini);
 public:
  static void Factory();
  static void ReFactory();
  static void DeFactory();
  static void TreeInit();
  static void ClearTree();
  static void SplitGini(int predIdx, int splitCount, int level);
};
#endif
