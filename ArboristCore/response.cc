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

/**
   @brief Regression-specific entry to factory methods.

   @param yNum is the response vector.

   @param levelMax is the current level-max value.

   @return void.
 */
void Response::FactoryReg(double *yNum, int levelMax) {
  nRow = Sample::NRow();
  response = new ResponseReg(yNum);
  SPReg::Factory(levelMax);
}

/**
   @base Classification-specific entry to factory methods.

   @param _yCtg is the response vector

   @param _perturb is a vector of numerical proxy values.

   @param levelMax is the current level-max value.

   @return cardinality of the response.
 */
// Requires an unadulterated zero-based version of the factor response as well as a
// clone subject to reordering.
//
int Response::FactoryCtg(const int _yCtg[], double perturb[], int levelMax) {
  nRow = Sample::NRow();

  int ctgWidth = ResponseCtg::Factory(_yCtg, perturb, levelMax);
  SPCtg::Factory(levelMax, ctgWidth);

  return ctgWidth;
}

/**
   @base Lights off initializations specific to classification.

   @paramm yCtg is the response vector.

   @param perturb is a vector of proxy values.

   @param levelMax is the current level-max value.

   @return cardinality of response.
*/
int ResponseCtg::Factory(const int _yCtg[], double perturb[], int levelMax) {
  treeJitter = new double[nRow];
  sumSquares = new double[levelMax];
  response =  new ResponseCtg(_yCtg, perturb, ctgWidth);
  ctgSum = new double[levelMax * ctgWidth];

  return ctgWidth;
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
   @brief Dummy error method to flag nonsensical categorical quantiles.
*/
void ResponseCtg::GetYRanked(double yRanked[]) {
  cout << "Quantile regression not supported for categorical response" << endl;
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
 @brief Constructor for categorical response.

 @param _yCtg is a stack temporary, so its contents must be copied and saved.

 @param perturb is the associated numerical proxy response.

 @param _ctgWidth outputs the response cardinality.
*/
ResponseCtg::ResponseCtg(const int _yCtg[], double perturb[], int &_ctgWidth) : Response(CtgFreq(_yCtg, perturb, _ctgWidth)) {
  yCtg = new int[nRow];
  for (int i = 0; i < nRow; i++)
    yCtg[i] = _yCtg[i];
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

/**
   @brief Computes relative frequencies of the various categories.

   @param yCtg is the categorical response vector.
   
   @param perturb is the proxy response vector.

   @param _ctgWidth outputs the response cardinality.

   @return Vector of corresponding frequencies, plus output parameter.
*/
// TODO:  Perhaps implemented more succinctly in front end.
double *ResponseCtg::CtgFreq(const int _yCtg[], double perturb[], int &_ctgWidth) {
  // Sorts a copy of the response, retaining the permutation induced.
  //
  int *yCtgClone = new int[nRow];
  for (int i = 0; i < nRow; i++) {    
    yCtgClone[i] = _yCtg[i];
  }

  int *perm = new int[nRow];
  for (int i = 0; i < nRow; i++) {
    perm[i] = i;
  }
  CallBack::QSortI(yCtgClone, perm, 1, nRow);

  // Walks sorted copy of '_yCtg[]' and over-writes with run cumulative
  // run lengths, from the second index up.
  //
  // '_ctgWidth' is assigned the number of distinct runs.
  //
  int yLast = yCtgClone[0];
  int runLength = 1;
  int numRuns = 1;
  for (int i = 1; i < nRow; i++) {
    int val = yCtgClone[i];
    if (yCtgClone[i] == yLast)
      runLength++;
    else {
      numRuns++;
      runLength = 1;
    }
    yLast = val;
    yCtgClone[i] = runLength;
  }
  if (numRuns < 2) // ASSERTION
    cout << "Single class tree" << endl;

  // Maintains a vector of compressed factors.
  //
  _ctgWidth = numRuns;
  double *yNum = new double[nRow]; // Freed by 'response' destructor.

  yLast = -1; // Dummy value to force new run on entry.
  // Scaled by conservative upper bound on interference:
  double recipRow = 1.0 / double(nRow);
  double scale = 0.5 * recipRow;
  for (int i = nRow - 1; i >= 0; i--) {
    if (yCtgClone[i] >= yLast) { // New run, possibly singleton:  resets length;
      runLength = yCtgClone[i];
    }
    yLast = yCtgClone[i];
    int idx = perm[i];
    // Assigns runlength frequency to the fractional part,
    // jittered by +- a value on a similar scale.
    yNum[idx] = recipRow * double(runLength) + scale * (perturb[i] - 0.5);
  }

  // Needs to be moved to a spot after 'nTree' is set.
  //  double treeBound = 2.0 * double(Train::nTree);
  //for (int i = 0; i < Train::nTree; i++)
  //treeJitter[i] = (perturb[i] - 0.5) / treeBound;

  delete [] perm;
  delete [] yCtgClone;

  return yNum;
  // Postcond:  numRuns == 0;
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
