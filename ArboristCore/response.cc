// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file response.cc

   @brief Methods maintaining the response-specific aspects of training.

   @author Mark Seligman

 */

#include "response.h"
#include "callback.h"
#include "sample.h"
#include "dectree.h"

//#include <iostream>
using namespace std;

unsigned int Response::nRow = 0;
Response *Response::response = 0;

unsigned int ResponseCtg::ctgWidth = 0;
int *ResponseCtg::yCtg = 0;

unsigned int *ResponseReg::row2Rank = 0;
double *ResponseReg::yRanked = 0;

void Response::FactoryReg(double yNum[]) {
  ResponseReg::Factory(yNum);
}

/**
   @brief Regression-specific entry to factory methods.

   @param yNum is the front end's response vector.

   @return void.
 */
void ResponseReg::Factory(double yNum[]) {
  nRow = Sample::NRow();
  response = new ResponseReg(yNum);
  SampleReg::Immutables();
}

/**
   @base Classification-specific entry to factory methods.

   @param feCtg is the response vector supplied by the front end.

   @param feProxy is a vector of numerical proxy values from the front end.

   @param ctgWidth is the cardinality of the response.

   @return void.
 */
// Requires an unadulterated zero-based version of the factor response as well as a
// clone subject to reordering.
//
void Response::FactoryCtg(const int feCtg[], const double feProxy[], unsigned int ctgWidth) {
  nRow = Sample::NRow();

  ResponseCtg::Factory(feCtg, feProxy, ctgWidth);
  SampleCtg::Immutables(ctgWidth);
}

/**
   @base Copies front-end vectors and lights off initializations specific to classification.

   @param feCtg is the front end's response vector.

   @param feProxy is the front end's vector of proxy values.

   @param ctgWidth is the cardinality of the response.

   @return void.
*/
void ResponseCtg::Factory(const int feCtg[], const double feProxy[], unsigned int _ctgWidth) {
  ctgWidth = _ctgWidth;
  yCtg = new int[nRow];
  double *_proxy = new double[nRow];
  for (unsigned int i = 0; i < nRow; i++) {    
    yCtg[i] = feCtg[i];
    _proxy[i] = feProxy[i];
  }
  response = new ResponseCtg(_proxy);
}

/**
 @brief Constructor for categorical response.

 @param _proxy is the associated numerical proxy response.

*/
ResponseCtg::ResponseCtg(double _proxy[]) : Response(_proxy) {
}

/**
   @brief Base class constructor.

   @param _y is the vector numerical/proxy response values.

 */
Response::Response(double _y[]) : y(_y) {
}

/**
   @brief Regression subclass constructor.

   @param _y is the response vector.
*/
ResponseReg::ResponseReg(double _y[]) : Response(_y) {
  // The only client is quantile regression, via Sample::sample2Rank[], but it is
  // simpler to compute in all cases and copy when needed.
  //
  row2Rank = new unsigned int[nRow];
  yRanked = new double[nRow];
  unsigned int *rank2Row = new unsigned int[nRow];
  for (unsigned int i = 0; i < nRow; i++) {
    yRanked[i] = _y[i];
    rank2Row[i] = i;
  }

  // Can implement rank as match(_y, sort(_y)) in Rcpp.
  CallBack::QSortD(yRanked, rank2Row, 1, nRow);
  for (unsigned int rk = 0; rk < nRow; rk++) {
    unsigned int row = rank2Row[rk];
    row2Rank[row] = rk;
  }
  delete [] rank2Row;
}

/**
   @brief Regression subclass destructor.
*/
ResponseReg::~ResponseReg() {
  delete [] row2Rank;
  delete [] yRanked;
  row2Rank = 0;
  yRanked = 0;
}


/**
 @brief Copies ranked response into caller's buffer.  Quantiles is the only client.

 @param _yRanked outputs the ranked response.

 @return void, with output vector parameter.
*/
void ResponseReg::GetYRanked(double _yRanked[]) {
  for (unsigned int i = 0; i < nRow; i++)
    _yRanked[i] = yRanked[i];
}

/**
 @brief Static pass-through for block tree sampling according to response type.

 @param inBag outputs whether row is in bag.

 @return count of in-bag rows, plus output param.
*/
Sample* Response::StageSamples(const PredOrd *predOrd, unsigned int inBag[], SamplePred *&samplePred, SplitPred *&splitPred, double &sum, int &bagCount) {
  return response->SampleRows(predOrd, inBag, samplePred, splitPred, sum, bagCount);
}


/**
   @brief Row-sampling entry for regression tree.

   @param inBag outputs whether row is in bag.

   @param 

   @return bag count for this tree.
*/
Sample *ResponseReg::SampleRows(const PredOrd *predOrd, unsigned int inBag[], class SamplePred *&samplePred, class SplitPred *&splitPred, double &sum, int &bagCount) {
  SampleReg *sampleReg = new SampleReg();

  bagCount = sampleReg->Stage(y, row2Rank, predOrd, inBag, samplePred, splitPred, sum);

  return sampleReg;
}


/**
   @brief Row-sampling entry for classification tree.

   @param inBag outputs whether row is in bag.

   @return Sample object, plus output param.
*/
Sample *ResponseCtg::SampleRows(const PredOrd *predOrd, unsigned int inBag[], class SamplePred *&samplePred, class SplitPred *&splitPred, double &sum, int &bagCount) {
  SampleCtg *sampleCtg = new SampleCtg();

  bagCount = sampleCtg->Stage(yCtg, y, predOrd, inBag, samplePred, splitPred, sum);

  return sampleCtg;
}


/**
   @brief Destructor for categorical response.
 */
ResponseCtg::~ResponseCtg() {
  delete [] y;
  delete [] yCtg;
  yCtg = 0;
}


/**
   @brief Outputs Info values of predictors and cleans up.

   @param predInfo is an output vector of predictor Info values.

   @return void, with output vector parameter.
 */
void Response::Finish(double predInfo[]) {
  DecTree::ScaleInfo(predInfo);
  DeFactorySt();
}


/**
   @brief Response entry for out-of-bag prediction.

   @param error outputs the prediction errors.

   @param predInfo is an output vector parameter reporting predictor Info values.

   @return void, with output parameter vector.
 */
void ResponseReg::PredictOOB(double error[], double predInfo[]) {
  DecTree::PredictAcrossReg(error, true);
  Finish(predInfo);
}


/**
   @brief Categorical entry for out-of-bag prediction.

   @param conf outputs a confusion matrix.

   @param error outputs the prediction errors.

   @param predInfo is an output vector parameter reporting predictor Info values.

   @param census records the predicted category counts, by row.

   @return void, with output parameters.
*/
void ResponseCtg::PredictOOB(int *conf, double error[], double predInfo[], int *census) {
  DecTree::PredictCtg(census, ctgWidth, yCtg, conf, error);

  Finish(predInfo);
}


/**
   @brief Static entry for finalizer.

   @response void.
 */
void Response::DeFactorySt() {
  response->DeFactory();
  delete response;
  response = 0;
}


/*
 */
void ResponseReg::DeFactory() {}

/**
   @brief Finalizer for categorical response.

   @return void.
 */
void ResponseCtg::DeFactory() {
  ctgWidth = 0;
}
