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
#include "predictor.h"
#include "sample.h"
#include "train.h"
#include "dectree.h"
#include "index.h"
#include "pretree.h"
#include "splitpred.h"

#include <iostream>
using namespace std;

int Response::bagCount = -1;
int Response::nRow = -1;
Response *Response::response = 0;

int ResponseCtg::ctgWidth = -1;
double *ResponseCtg::treeJitter = 0;
double *ResponseCtg::ctgSum = 0;
double *ResponseCtg::sumSquares = 0;
int *ResponseCtg::yCtg = 0;

int *ResponseReg::sample2Rank = 0;
int *ResponseReg::row2Rank = 0;
double *ResponseReg::yRanked = 0;


void Response::FactoryReg(double yNum[], int levelMax) {
  ResponseReg::Factory(yNum, levelMax);
}
/**
   @brief Regression-specific entry to factory methods.

   @param yNum is the front end's response vector.

   @param levelMax is the current level-max value.

   @return void.
 */
void ResponseReg::Factory(double yNum[], int levelMax) {
  nRow = Sample::NRow();
  response = new ResponseReg(yNum);
  SPReg::Factory(levelMax);
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
void Response::FactoryCtg(const int feCtg[], const double feProxy[], int ctgWidth, int levelMax) {
  nRow = Sample::NRow();

  ResponseCtg::Factory(feCtg, feProxy, ctgWidth, levelMax);
  SPCtg::Factory(levelMax, ctgWidth);
}

/**
   @base Copies front-end vectors and lights off initializations specific to classification.

   @param feCtg is the front end's response vector.

   @param feProxy is the front end's vector of proxy values.

   @param ctgWidth is the cardinality of the response.

   @param levelMax is the current level-max value.

   @return void.
*/
void ResponseCtg::Factory(const int feCtg[], const double feProxy[], int _ctgWidth, int levelMax) {
  ctgWidth = _ctgWidth;
  treeJitter = new double[nRow];
  sumSquares = new double[levelMax];
  yCtg = new int[nRow];
  double *_proxy = new double[nRow];
  for (int i = 0; i < nRow; i++) {    
    yCtg[i] = feCtg[i];
    _proxy[i] = feProxy[i];
  }
  response = new ResponseCtg(_proxy);
  ctgSum = new double[levelMax * ctgWidth];
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
  row2Rank = new int[nRow];
  yRanked = new double[nRow];
  int *rank2Row = new int[nRow];
  for (int i = 0; i < nRow; i++) {
    yRanked[i] = _y[i];
    rank2Row[i] = i;
  }

  // Can implement rank as match(_y, sort(_y)) in Rcpp.
  CallBack::QSortD(yRanked, rank2Row, 1, nRow);
  for (int rk = 0; rk < nRow; rk++) {
    int row = rank2Row[rk];
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
  for (int i = 0; i < nRow; i++)
    _yRanked[i] = yRanked[i];
}

/**
 @brief Initializes per-tree member.

 @param levelMax is the current level-max value.

 @return In-bag size of current tree.
*/
int Response::SampleRows(int levelMax) {
  int *rvRows = new int[nRow];
  CallBack::SampleRows(rvRows);
  bagCount = response->SampleRows(rvRows);
  delete [] rvRows;

  PreTree::TreeInit(levelMax, bagCount);
  double sum = response->Sum();
  IndexNode::TreeInit(levelMax, bagCount, Sample::NSamp(), sum);
  response->TreeInit();

  return bagCount;
}

/**
   @brief Row-sampling entry for regression tree.

   @param rvRows is the vector of sampled row indices.

   @return in-bag size of sampled row indices.
*/
int ResponseReg::SampleRows(const int rvRows[]) {
  return SampleReg::SampleRows(rvRows, y, row2Rank);
}

/**
   @brief Row-sampling entry for classification tree.

   @param rvRows is the vector of sampled row indices.

   @param yCtg is the categorical response.

   @param y is the proxy response vector.

   @return in-bag sample size.
*/
int ResponseCtg::SampleRows(const int rvRows[]) {
  return SampleCtg::SampleRows(rvRows, yCtg, y);
}

/**
   @brief Destructor for categorical response.
 */
ResponseCtg::~ResponseCtg() {
  delete [] y;
  delete [] yCtg;
  delete [] treeJitter;
  yCtg = 0;
  treeJitter = 0;
}

double ResponseCtg::Jitter(int row) {
  return 0.0;//jitter[row];
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

   @return void, with output parameters.
*/
void ResponseCtg::PredictOOB(int *conf, double error[], double predInfo[]) {
  DecTree::PredictAcrossCtg(yCtg, ctgWidth, conf, error);
  Finish(predInfo);
}

/**
   @brief Regression-specific per-tree initializations.

   @return void.
*/
void ResponseReg::TreeInit() {
  SampleReg::Stage();
}

/**
   @brief Regression-specific sample summation

   @return sum of in-bag response values.
*/
double ResponseReg::Sum() {
  return SampleReg::Sum(bagCount);
}

/**
   @brief Categorical-specifi sample summation.

   @return sum of in-bag response values.
 */
double ResponseCtg::Sum() {
  return SampleCtg::Sum(bagCount);
}

/**
   @brief Categorical-specific staging of sampled predictors.

   @return void.
*/
void ResponseCtg::TreeInit() {
  SampleCtg::Stage();
}

/**
   @brief Static entry point for tree finalizers.

   @return void.
 */
void Response::TreeClearSt() {
  response->TreeClear();
  IndexNode::TreeClear();
  PreTree::TreeClear();
  bagCount = -1;
}

/**
   @brief Regression-specific tree finalizer.

   @return void.
 */
void ResponseReg::TreeClear() {
  SampleReg::TreeClear();
}

/**
   @brief Categorical-specific tree finalizer.

   @return void.
*/
void ResponseCtg::TreeClear() {
  SampleCtg::TreeClear();
}

/**
   @brief Static entry for level reallocation.

   @return void.
*/
void Response::ReFactory(int levelMax) {
  response->ReFactorySP(levelMax);
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

/**
   @brief Static entry for level reallocation.

   @param levelMax is the current level-max value.

   @return void.
 */
void ResponseReg::ReFactorySP(int levelMax) {
  SPReg::ReFactory(levelMax);
}

/**
   @brief Finalizer for regression response.

   @return void.
 */
void ResponseReg::DeFactory() {
  SPReg::DeFactory();
}


/**
   @brief Static entry for categorical reallocation.

   @param levelMax is the current level-max value.

   @return void.
 */
void ResponseCtg::ReFactorySP(int levelMax) {
  delete [] ctgSum;
  delete [] sumSquares;
  ctgSum = new double[levelMax * ctgWidth];
  sumSquares = new double[levelMax];

  SPCtg::ReFactory(levelMax);
}

/**
   @brief Finalizer for categorical response.

   @return void.
 */
void ResponseCtg::DeFactory() {
  delete [] ctgSum;
  delete [] sumSquares;
  ctgSum = 0;
  sumSquares = 0;
  SPCtg::DeFactory();
  ctgWidth = -1;
}

/**
   @brief Invokes response-specific sum methods.

   @return void.
 */
void Response::LevelSums(int splitCount) {
  response->Sums(splitCount);
}

/**
 @brief Initializes 'ctgSum[]' and 'sumSquares[]' values for nodes making
 it to the next level.

 These must be computed in order to set the pre-bias. Next level's split
 nodes should be in place before invocation, so that values are
 indexable by split position.  Similarly, Replay() invocations should have taken
 place, as well.

 @param splitCount is the number of splits in the next level.

 @return void.
*/
void ResponseCtg::Sums(int splitCount) {
  int levelWidth = PreTree::LevelWidth();
  double *sumTemp = new double[levelWidth * ctgWidth];
  for (int i = 0; i < levelWidth * ctgWidth; i++)
    sumTemp[i] = 0.0;

  // Sums each category for each node in the upcoming level, including
  // leaves.  Since these appear in arbitrary order, a second pass copies
  // those columns corresponding to nonterminals in split-index order.
  //
  for (int sIdx = 0; sIdx < bagCount; sIdx++) {
    int levelOff = PreTree::LevelSampleOff(sIdx);
    if (levelOff >= 0) {
      double sum;
      int ctg = SampleCtg::CtgSum(sIdx, sum);
      sumTemp[levelOff * ctgWidth + ctg] += sum;
    }
  }

  // Reorders by split index and compresses away any intervening leaf sums.
  // Could index directly by level offset instead, but this would require
  // more complex accessor methods.
  //
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    int levelOff = PreTree::LevelOff(IndexNode::PTId(splitIdx));
    double ss = 0.0;
    for (int ctg = 0; ctg < ctgWidth; ctg++) {
      double sum = sumTemp[ctgWidth * levelOff + ctg];
      ctgSum[ctgWidth * splitIdx + ctg] = sum;
      ss += sum * sum;
    }
    sumSquares[splitIdx] = ss;
  }
  delete [] sumTemp;
}

/**
   @brief Dummy method for regression response.
 */
void ResponseReg::Sums(int splitCount) {
}

/**
   @brief Static entry for pre-bias computation.

   @param splitIdx is the split index.

   @return pre-bias value for the split.
 */
double Response::PrebiasSt(int splitIdx) {
  return response->Prebias(splitIdx);
}

/**
  @brief Pre-bias computation for regression response.

  @param splitIdx is the split index.

  @return pre-bias value.
*/
double ResponseReg::Prebias(int splitIdx) {
  double sum;
  int sCount;
  IndexNode::PrebiasFields(splitIdx, sum, sCount);
  return (sum * sum) / sCount;
}


/**
   @brief Pre-bias computation for categorical response.

   @param splitIdx is the split index.

   @see ResponseCtg::Sums

   @return pre-bias value.
 */
double ResponseCtg::Prebias(int splitIdx) {
  double sum;
  int sCount;
  IndexNode::PrebiasFields(splitIdx, sum, sCount);
  return sumSquares[splitIdx] / sum;
}

/**
   @brief Virtual entry for score computation.

   @param treeHeight is the number of indices in the pretree.

   @param scores outputs the computed score values.

   @return void, with output parameter vector.
 */
void Response::ProduceScores(int treeHeight, double scores[]) {
  response->Scores(treeHeight, scores);
}


/**
   @brief Regression entry for score computation.

   @param treeHeight is the number of indices in the pretree.

   @param scores outputs the computed score values.

   @return void, with output parameter vector.
 */
void ResponseReg::Scores(int treeHeight, double scores[]) {
  SampleReg::Scores(bagCount, treeHeight, scores);
}

/**
   @brief Categorical entry for score computation.

   @param treeHeight is the number of indices in the pretree.

   @param scores outputs the computed score values.

   @return void, with output parameter vector.
 */
void ResponseCtg::Scores(int treeHeight, double scores[]) {
  SampleCtg::Scores(bagCount, ctgWidth, treeHeight, scores);
}
