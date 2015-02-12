// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file response.h

   @brief Class definitions for representing response-specific aspects of training, especially regression versus categorical support.

   @author Mark Seligman

 */


#ifndef ARBORIST_RESPONSE_H
#define ARBORIST_RESPONSE_H

/**
   @brief Methods and members for management of response-related computations.
 */
class Response {
 protected:
  static int bagCount;
  static void Finish(double predInfo[]);
 public:
  double *y;
  static int nRow; // Set from Predictor
  static Response *response;
  static void FactoryReg(double yNum[], int levelMax);
  static void FactoryCtg(const int feCtg[], const double feProxy[], int ctgWidth, int levelMax);
  static void DeFactorySt();
  virtual void DeFactory() = 0;
  Response(double _y[]);
  static int SampleRows(int levelMax);
  virtual int SampleRows(const int rvRows[]) = 0;
  virtual double Sum() = 0;
  virtual void TreeInit() = 0;
  static void TreeClearSt();
  virtual void TreeClear() = 0;
  static void ProduceScores(int leafCount, double scores[]);
  virtual void Scores(int leafCount, double scores[]) = 0;
  static void ReFactory(int levelMax);
  virtual void ReFactorySP(int levelMax) = 0;
  static void LevelSums(int splitCount);
  virtual void Sums(int splitCount) = 0;
  virtual ~Response(){}
  static double PrebiasSt(int splitIdx);
  virtual double Prebias(int splitIdx) = 0;
};

/**
   @brief Specialization to regression trees.
 */
class ResponseReg : public Response {
  static int *sample2Rank;
 public:
  ResponseReg(double _y[]);
  ~ResponseReg();
  static void Factory(double yNum[], int levelMax);
  static int *row2Rank;
  static double *yRanked;
  void Scores(int leafCount, double scores[]);
  static void PredictOOB(double err[], double predInfo[]);
  static void PredictOOB(double error[], double quantVec[], int qCells, double qPred[], double predInfo[]);
  static void GetYRanked(double _yRanked[]);
  void ReFactorySP(int levelMax);
  int SampleRows(const int rvRows[]);
  static void ReFactory();
  void DeFactory();
  void TreeInit();
  double Sum();
  void TreeClear();
  void Sums(int splitNext);
  double Prebias(int splitIdx);
};

/**
   @brief Specialization to classification trees.
 */
class ResponseCtg : public Response {
  static int ctgWidth;
  static double *treeJitter; // Helps prevent ties among response scores.
  static double *ctgSum; // [#splits]:  Re-allocatable
  static double *sumSquares; // [#splits]:  Re-allocatable
 public:
  static void Factory(const int feCtg[], const double feProxy[], int ctgWidth, int levelMax);
  static int *yCtg; // The original factor-valued response.

  ResponseCtg(double yProxy[]);
  static double Jitter(int row);
  static void PredictOOB(int *conf, double err[], double predInfo[]);
  ~ResponseCtg();
  void ReFactorySP(int levelMax);
  static void Factory(int _yCtg[], int _ctgWidth, int levelMax);
  static void ReFactory();
  void DeFactory();
  void TreeClear();
  void TreeInit();
  double Sum();
  int SampleRows(const int rvRows[]);
  void Scores(int leafCount, double scores[]);

  double Prebias(int splitIdx);
  void Sums(int splitNext);

  // Splitting and Prebias methods are the only clients.
  //
  static inline double SumSquares(int splitIdx) {
    return sumSquares[splitIdx];
  }

  static inline double CtgSum(int splitIdx, int ctg) {
    return ctgSum[splitIdx * ctgWidth + ctg];
  }
};

#endif
