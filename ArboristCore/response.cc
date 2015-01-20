// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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

void Response::FactoryReg(double *yNum, int levelMax) {
  nRow = Sample::NRow();
  // assert(nRow > 0):  Factory ordering.
  response = new ResponseReg(yNum);
  SPReg::Factory(levelMax);
}

// Requires an unadulterated zero-based version of the factor response as well as a
// clone subject to reordering.
//
int Response::FactoryCtg(const int _yCtg[], double perturb[], int levelMax) {
  nRow = Sample::NRow();
  // assert(nRow > 0):  Factory ordering.

  int ctgWidth = ResponseCtg::Factory(_yCtg, perturb, levelMax);
  SPCtg::Factory(levelMax, ctgWidth);

  return ctgWidth;
}

int ResponseCtg::Factory(const int _yCtg[], double perturb[], int levelMax) {
  treeJitter = new double[nRow];
  sumSquares = new double[levelMax];
  response =  new ResponseCtg(_yCtg, perturb, ctgWidth);
  ctgSum = new double[levelMax * ctgWidth];

  return ctgWidth;
}

Response::Response(double _y[]) : y(_y) {
}

// TODO: sort 'y' for quantile regression.
//
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


ResponseReg::~ResponseReg() {
  delete [] row2Rank;
  delete [] yRanked;
  row2Rank = 0;
  yRanked = 0;
}

// Front end should ensure that this path is not reached.
//
void ResponseCtg::GetYRanked(double yRanked[]) {
  cout << "Quantile regression not supported for categorical response" << endl;
}

// Copies 'yRanked' into caller's buffer.  Quantiles is the only client.
//
void ResponseReg::GetYRanked(double _yRanked[]) {
  for (int i = 0; i < nRow; i++)
    yRanked[i] = yRanked[i];
}

// '_yCtg' is a stack temporary, so its contents must be copied and saved.
//
ResponseCtg::ResponseCtg(const int _yCtg[], double perturb[], int &_ctgWidth) : Response(CtgFreq(_yCtg, perturb, _ctgWidth)) {
  yCtg = new int[nRow];
  for (int i = 0; i < nRow; i++)
    yCtg[i] = _yCtg[i];
}

// Sets 'bagCount' for the current tree and initializes per-tree instances.
//
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

int ResponseReg::SampleRows(const int rvRows[]) {
  return SampleReg::SampleRows(rvRows, y, row2Rank);
}

int ResponseCtg::SampleRows(const int rvRows[]) {
  return SampleCtg::SampleRows(rvRows, yCtg, y);
}

ResponseCtg::~ResponseCtg() {
  delete [] y;
  delete [] yCtg;
  delete [] treeJitter;
  yCtg = 0;
  treeJitter = 0;
}

// Returns vector of relative frequency of categories at corresponding index in
// 'yCtgClone'.
//
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

void ResponseReg::PredictOOB(int *conf, double error[]) {
  DecTree::PredictAcrossReg(error);
}

void ResponseCtg::PredictOOB(int *conf, double error[]) {
  DecTree::PredictAcrossCtg(yCtg, ctgWidth, conf, error);
}

// Returns sum of samples for tree.
//
void ResponseReg::TreeInit() {
  SampleReg::Stage();
}

double ResponseReg::Sum() {
  return SampleReg::Sum(bagCount);
}

double ResponseCtg::Sum() {
  return SampleCtg::Sum(bagCount);
}

// 'ctgSum' is allocated per-session, so must be reinitialized on tree entry.
//
void ResponseCtg::TreeInit() {
  SampleCtg::Stage();
  SPCtgFac::TreeInit();
}

void Response::TreeClearSt() {
  response->TreeClear();
  IndexNode::TreeClear();
  PreTree::TreeClear();
  bagCount = -1;
}

void ResponseReg::TreeClear() {
  SampleReg::TreeClear();
}

void ResponseCtg::TreeClear() {
  SampleCtg::TreeClear();
}

  /* ASSERTIONS:
  if (abs(sum - sCount) > 0.1)
    cout << "Jitter mismatch" << endl;
  double sumByCtg = 0.0;
  for (int i = 0; i < LevelCtg::ctgWidth; i++) {
    sumByCtg += sumBase[i];
  }
  if (abs(sumByCtg - sum) > 0.1)
    cout << sumByCtg - sum << endl;
  */

void Response::ReFactory(int levelMax) {
  response->ReFactorySP(levelMax);
}

void Response::DeFactorySt() {
  response->DeFactory();
  delete response;
  response = 0;
}

//
void ResponseReg::ReFactorySP(int levelMax) {
  SPReg::ReFactory(levelMax);
}

void ResponseReg::DeFactory() {
  SPReg::DeFactory(); // TODO:  Move to Index, as this caller is invoked at Predict::
}


//
void ResponseCtg::ReFactorySP(int levelMax) {
  delete [] ctgSum;
  delete [] sumSquares;
  ctgSum = new double[levelMax * ctgWidth];
  sumSquares = new double[levelMax];

  SPCtg::ReFactory(levelMax);
}

void ResponseCtg::DeFactory() {
  delete [] ctgSum;
  delete [] sumSquares;
  ctgSum = 0;
  sumSquares = 0;
  SPCtg::DeFactory();
  ctgWidth = -1;
}

void Response::LevelSums(int splitCount) {
  response->Sums(splitCount);
}

// Initializes 'ctgSum[]' and 'sumSquares[]' values for nodes making
// it to the next level.  These must be computed in order to set the pre-bias.
//
// Next level's split nodes should be in place before invocation, so that values are
// indexable by split position.  Similarly, Replay() invocations should have taken
// place, as well.
//
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

void ResponseReg::Sums(int splitCount) {
}

double Response::PrebiasSt(int splitIdx) {
  return response->Prebias(splitIdx);
}

double ResponseReg::Prebias(int splitIdx) {
  double sum;
  int sCount;
  IndexNode::PrebiasFields(splitIdx, sum, sCount);
  return (sum * sum) / sCount;
}


//
double ResponseCtg::Prebias(int splitIdx) {
  double sum;
  int sCount;
  IndexNode::PrebiasFields(splitIdx, sum, sCount);
  return sumSquares[splitIdx] / sum;
}

void Response::ProduceScores(int treeHeight, double scores[]) {
  response->Scores(treeHeight, scores);
}

void ResponseReg::Scores(int treeHeight, double scores[]) {
  SampleReg::Scores(bagCount, treeHeight, scores);
}

void ResponseCtg::Scores(int treeHeight, double scores[]) {
  SampleCtg::Scores(bagCount, ctgWidth, treeHeight, scores);
}

void Response::DispatchQuantiles(int treeSize, int leafPos[], int leafExtent[], int rank[], int rankCount[]) {
  response->Quantiles(treeSize, leafPos, leafExtent, rank, rankCount);
}

void ResponseCtg::Quantiles(int treeSize, int leafPos[], int leafExtent[], int rank[], int rankCount[]) {
  cout << "Quantiles undefined for categorical response" << endl;
  // error()
}

void ResponseReg::Quantiles(int treeSize, int leafPos[], int leafExtent[], int rank[], int rankCount[]) {
  SampleReg::DispatchQuantiles(treeSize, bagCount, leafPos, leafExtent, rank, rankCount);
}



