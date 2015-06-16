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
  static void Finish(double predInfo[]);
 public:
  double *y;
  static unsigned int nRow; // Set from Predictor
  static Response *response;
  static void FactoryReg(double yNum[]);
  static void FactoryCtg(const int feCtg[], const double feProxy[], unsigned int ctgWidth);
  static void DeFactorySt();
  virtual void DeFactory() = 0;
  Response(double _y[]);
  static class Sample* StageSamples(const class PredOrd *predOrd, unsigned int inBag[], class SamplePred *&samplePred, class SplitPred *&splitPred, double &sum, int &bagCount);
  virtual Sample* SampleRows(const class PredOrd *predOrd, unsigned int inBag[], class SamplePred *&samplePred, class SplitPred *&splitPred, double &sum, int &bagCount) = 0;
  
  virtual ~Response(){}
};

/**
   @brief Specialization to regression trees.
 */
class ResponseReg : public Response {
 public:
  ResponseReg(double _y[]);
  ~ResponseReg();
  static void Factory(double yNum[]);
  static unsigned int *row2Rank;
  static double *yRanked;
  static void PredictOOB(double err[], double predInfo[]);
  static void PredictOOB(double error[], double quantVec[], int qCells, double qPred[], double predInfo[]);
  static void GetYRanked(double _yRanked[]);
  Sample* SampleRows(const class PredOrd *predOrd, unsigned int inBag[], class SamplePred *&samplePred, class SplitPred *&splitPred, double &sum, int &bagCount);
  void DeFactory();
};

/**
   @brief Specialization to classification trees.
 */
class ResponseCtg : public Response {
  static unsigned int ctgWidth; // Fixed by response.
  static double *treeJitter; // Helps prevent ties among response scores.
 public:
  static void Factory(const int feCtg[], const double feProxy[], unsigned int ctgWidth);
  static int *yCtg; // The original factor-valued response.

  ResponseCtg(double yProxy[]);
  static double Jitter(int row);
  static void PredictOOB(int *conf, double err[], double predInfo[]);
  ~ResponseCtg();
  static void Factory(int _yCtg[], unsigned int _ctgWidth);
  void DeFactory();
  Sample* SampleRows(const class PredOrd *predOrd, unsigned int inBag[], class SamplePred *&samplePred, class SplitPred *&splitPred, double &sum, int &bagCount);
  static int CtgSum(unsigned int sIdx, double &sum);
};

#endif
