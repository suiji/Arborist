/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "response.h"
#include "predictor.h"
#include "dataord.h"
#include "train.h"
#include "dectree.h"
#include "node.h"
#include "pretree.h"

//#include <R.h>
//#include <Rcpp.h> // TODO:  Exit sorting dependencies.
//using namespace Rcpp;

#include <iostream>
using namespace std;

Response *Response::response = 0;
int ResponseCtg::ctgWidth = -1;
int ResponseCtg::leafWSHeight = -1;
double *ResponseCtg::leafWS = 0;
double *ResponseCtg::treeJitter = 0;

int *ResponseReg::row2Rank = 0;
double *ResponseReg::ySorted = 0;

void Response::Factory(double *yNum, int _nRow) {
  nRow = _nRow;
  response = new ResponseReg(yNum);
}

// Requires an unadulterated zero-based version of the factor response as well as a
// clone subject to reordering.
//
int Response::Factory(const int _yCtg[], double perturb[], int _nRow) {
  int ctgWidth;

  nRow = _nRow;
  ResponseCtg::Factory(_yCtg, perturb, ctgWidth);

  return ctgWidth;
}

void ResponseCtg::Factory(const int _yCtg[], double perturb[], int &_ctgWidth) {
  treeJitter = new double[nRow];
  response =  new ResponseCtg(_yCtg, perturb, _ctgWidth);

  leafWS = new double[_ctgWidth * Train::nSamp]; 
}

Response::Response(double *_y) : y(_y) {
}

// TODO: sort 'y' for quantile regression.
//
ResponseReg::ResponseReg(double *_y) : Response(_y) {

  // The only client is quantile regression, via Node::sample2Rank, but it is
  // simpler to compute in all cases and copy when needed.
  //
  row2Rank = new int[nRow];
  ySorted = new double[nRow];
  int *rank2Row = new int[nRow];
  for (int i = 0; i < nRow; i++) {
    ySorted[i] = _y[i];
    rank2Row[i] = i;
  }

  // If a rank function were available, it would not be necesary to clone and
  // sort 'y'.
  //
  R_qsort_I(ySorted, rank2Row, 1, nRow);
  for (int rk = 0; rk < nRow; rk++) {
    int row = rank2Row[rk];
    row2Rank[row] = rk;
  }

  delete [] rank2Row;
}


ResponseReg::~ResponseReg() {
  delete [] row2Rank;
  delete [] ySorted;
  row2Rank = 0;
  ySorted = 0;
}

// Front end should ensure that this path is not reached.
//
void ResponseCtg::GetYRanked(double yRanked[]) {
  cout << "Quantile regression not supported for categorical response" << endl;
}

// Copies 'ySorted' into caller's buffer.
//
void ResponseReg::GetYRanked(double yRanked[]) {
  for (int i = 0; i < nRow; i++)
    yRanked[i] = ySorted[i];
}

// '_yCtg' is a stack temporary, so its contents must be copied and saved.
//
ResponseCtg::ResponseCtg(const int _yCtg[], double perturb[], int &_ctgWidth) : Response(FactorFreq(_yCtg, perturb)) {
  yCtg = new int[nRow];
  for (int i = 0; i < nRow; i++)
    yCtg[i] = _yCtg[i];
  leafWS = 0;
  leafWSHeight = -1;
  _ctgWidth = ctgWidth;
}

// Sets 'bagCount' for the current tree.
//
int Response::SampleRows(int nRow) {
  int *rvRows = sampReplace ? SampleWith(nSamp) : SampleWithout(rowVec.begin(), rowVec.length(), nSamp);

  return response->SampleRows(rvRows, nRow);
}

int ResponseReg::SampleRows(const int rvRows[], int nRow) {
  return DataOrd::SampleRows(sampleRows, sample, sample2Rank);
}

int ResponseCtg::SampleRows(const int rvRows[], int nRow) {
  return DataOrd::SampleRows(sampleRows, nRow, yCtg, y, sampleCtg);
}

ResponseCtg::~ResponseCtg() {
  delete [] y;
  delete [] leafWS;
  delete [] yCtg;
  delete [] treeJitter;
  leafWS = 0;
  yCtg = 0;
  treeJitter = 0;
}

void Response::DeFactory() {
  delete response;
  response = 0;
}

// Returns vector of relative frequency of factors at corresponding index in
// 'yCtgClone'.
//
double *ResponseCtg::FactorFreq(const int _yCtg[], double perturb[]) {
  int *yCtgClone = new int[nRow];
  for (int i = 0; i < nRow; i++) {    
    yCtgClone[i] = _yCtg[i];
  }

  int perm[nRow];
  for (int i = 0; i < nRow; i++) {
    perm[i] = i;
  }
  R_qsort_int_I(yCtgClone, perm, 1, nRow);

  // Overwrites yCtgClone[] with current runlength.
  //
  int yLast = yCtgClone[0];
  int runLength = 1;
  int numRuns = 1;
  //int sum = 0;
  for (int i = 1; i < nRow; i++) {
    int val = yCtgClone[i];
    //sum++;
    if (yCtgClone[i] == yLast)
      runLength++;
    else {
      numRuns++;
      runLength = 1;
    }
    yLast = val;
    yCtgClone[i] = runLength;
  }
  if (numRuns < 2)
    cout << "Single class tree" << endl;

  // Maintains a vector of compressed factors.
  //
  ctgWidth = numRuns;
  double *yNum = new double[nRow]; // Freed by 'response' destructor.

  yLast = -1; // Dummy value to force new run on entry.
  // Conservative upper bound on interference.
  double bound = double(nRow) * 2.0;
  for (int i = nRow - 1; i >= 0; i--) {
    if (yCtgClone[i] >= yLast) { // New run, possibly singleton:  resets length;
      runLength = yCtgClone[i];
    }
    yLast = yCtgClone[i];
    int idx = perm[i];
    // Assigns normalized value, jittered by +- a value on a similar scale.
    yNum[idx] = 1.0 /* * double(runLength) / nRow*/ + (perturb[i] - 0.5) / bound;
  }

  //  for (int i = 0; i < nRow; i++)
  //cout << yNum[i] << " / " << _yCtg[i] << endl;

  // Needs to be moved to a spot after 'nTree' is set.
  //  double treeBound = 2.0 * double(Train::nTree);
  //for (int i = 0; i < Train::nTree; i++)
  //treeJitter[i] = (perturb[i] - 0.5) / treeBound;

  //  cout << "Dividing by " << sum << endl;
  //for (int i = 0; i < ln;i++)
  //cout << perm[i] << ":  " << yNum[perm[i]] << endl;
  delete [] yCtgClone;

  return yNum;
  // Postcond:  numRuns == 0;
}

double ResponseCtg::Jitter(int row) {
  return 0.0;//jitter[row];
}


// Scores are extracted once per tree, after all leaves have been marked.
//
void ResponseCtg::ProduceScores(const int *sample2Accum, const SampleCtg sampleCtg[]) {
  for (int i = 0; i < ctgWidth * Train::nSamp; i++)
    leafWS[i] = 0.0;

  // Irregular access.  Needs the ability to map sample indices to the factors and
  // weights with which they are associated.  This can be achieved by mapping samples
  // to original row numbers, then indexing into the respective response vectors.
  //
  for (int i = 0; i < AccumHandler::bagCount; i++) {
    int leafIdx = sample2Accum[i]; // Index into leaf set. (was absolute tree offset)
    int ctg = sampleCtg[i].ctg;
    // ASSERTION:
    if (ctg < 0 || ctg >= ctgWidth)
      cout << "Bad response category:  " << ctg << endl;
    // ASSERTION:
    if (leafIdx < 0)
      cout << "Untreated index" << endl;
    double responseWeight = sampleCtg[i].val;
    leafWS[ctg + leafIdx * ctgWidth] += responseWeight;
  }

  // Factor weights have been jittered, making ties highly unlikely.  Even in the
  // event of a tie, although the first in the run is chosen, the jittering itself
  // is nondeterministic.
  //
  // Every leaf should obtain a non-negative factor-valued score.
  //
  for (int leafIdx = 0; leafIdx < PreTree::leafCount; leafIdx++) {
    // Assertion
    //    if (leafIdx < 0)
    //cout << "Negative leaf tag" << endl;

    double *ctgBase = leafWS + leafIdx * ctgWidth;
    double maxWeight = 0.0;
    int maxWeightIdx = -1;
    for (int ctg = 0; ctg < ctgWidth; ctg++) {
      double thisWeight = ctgBase[ctg];
      //cout << "Leaf " << leafIdx << " factor index: " << fac << ", weight:  " << thisWeight << endl;
      if (thisWeight > maxWeight) {
	maxWeight = thisWeight;
	maxWeightIdx = ctg;
      }
    }
    // ASSERTION:
    if (maxWeightIdx == -1)
      cout << "Scoreless leaf " << leafIdx << endl;
    PreTree::leafSet[leafIdx]->score = maxWeightIdx; // For now, upcasts score to double, for compatability with DecTree.
    //    cout << leafIdx << ":  " << maxWeightIdx << endl;
  }
}

void Response::NodeFactory(int &auxSize) {
  response->Node(auxSize);
}

void ResponseReg::Nodes(int &auxSize) {
  NodeReg::Factory();
}

void ResponseCtg::Nodes(int &auxSize) {
  Node::Factory(yCtg, ctgWidth, auxSize);
}

void ResponseReg::PredictOOB(int *conf, double error[]) {
  DecTree::PredictAcrossReg(error);
}

void ResponseCtg::PredictOOB(int *conf, double error[]) {
  DecTree::PredictAcrossCtg(yCtg, ctgWidth, conf, error);
}
