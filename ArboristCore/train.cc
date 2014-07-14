/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "dectree.h"
#include "dataord.h"
#include "train.h"
#include "predictor.h"
#include "dectree.h"
#include "node.h"
#include "response.h"
#include "splitsig.h"
#include "math.h"

#include <iostream>
using namespace std;


int Train::accumRealloc = -1;
int Train::probResize = -1;
int Train::nTree = -1;
int Train::nSamp = -1;
int Train::qCells = -1;
double *Train::qVec = 0;
bool Train::sampReplace = true;
bool Train::doQuantiles = false;
double Train::minRatio = 0.0;
int Train::blockSize = -1;
int *Train::cdfOff = 0;
double *Train::sCDF = 0;

// Singleton factory:  everything is static.
//
void Train::Factory(int _nTree, int _nSamp, bool _sampReplace, bool _quantiles, double _minRatio, int _blockSize) {
  nTree = _nTree;
  nSamp = _nSamp;
  sampReplace = _sampReplace;
  doQuantiles = _quantiles;
  blockSize = _blockSize;
  minRatio = _minRatio;
  accumRealloc = 0;
  probResize = 0;
}

void Train::DeFactory() {
  //  cout << probResize << " prob resizes, " << accumRealloc << " accum reallocations" << endl;
  if (cdfOff != 0) {
    delete [] cdfOff;
    delete [] sCDF;
    cdfOff = 0;
    sCDF = 0;
  }
    
}

/*
int *Train::SampleWith(int ln) {
  IntegerVector targ(ln);
  RNGScope scope;
  NumericVector rn(runif(ln));
  if (sCDF == 0) {
    for (int i = 0; i < ln; i++) {
      targ[i] = ln * rn[i];
    }
  }
  else {
    for (int i = 0; i < ln; i++) {
      targ[i] = SampleDraw(rn[i]);
    }
  }
      //    if (targ[i] >= ln || targ[i] <0)
    //cout << "OVERFLOW" << endl;

  return targ.begin();
}

// Sampling without replacement of 0 through 'ln'-1.
//
int *Train::SampleWithout(const int sampVec[], int sourceLen, int targLen) {
  IntegerVector targ(targLen);
  int *swopVec = new int[sourceLen];
  for (int i = 0; i < sourceLen;i++)
    swopVec[i] = sampVec[i];
  int top = sourceLen - 1;
  RNGScope scope;
  NumericVector rn(runif(targLen));
  //  IntegerVector rv = floor(seq_len(targeLen) * rn);
  for (int i = 0; i < targLen; top--, i++) {
    int idx = floor(rn[i] * top);
    int temp = swopVec[idx];
    swopVec[idx] = swopVec[top];
    swopVec[top] = temp;
    //    cout << temp << endl;
    targ[i] = temp;
  }
  delete [] swopVec;

  return targ.begin();
}
*/

void Train::IntBlock(int xBlock[], int _nrow, int _ncol) {
  Predictor::IntegerBlock(xBlock, _nrow, _ncol);
}

void Train::ResponseReg(double y[]) {
  Response::Factory(y);
}

int Train::ResponseCtg(const int y[], double yPerturb[]) {
  return Response::Factory(y, yPerturb);
}

// Writes a scaled weighting vector into 'predWeight' from the raw weights in 'sPredWeight'.
// Conditions these weights on the overall probabilty of a predictor, 'predProb'.
//
void Train::PredWeight(const double weights[], const double predProb) {
  double *predWeight = new double[Predictor::nPred];
  double max = 0.0;

  for (int i = 0; i < Predictor::nPred; i++) {
    double wt = weights[i];
    predWeight[i] = wt;
    max = max > wt ? max : wt;
  }

  double scaleWeight = predProb / max;
  for (int i = 0; i < Predictor::nPred; i++) {
    predWeight[i] *= scaleWeight;
  }

  Predictor::SetProbabilities(predWeight);
  delete [] predWeight;
}

void Train::TrainInit(double *_predWeight, double _predProb, int _nTree, int _nSamp, bool _smpReplace, bool _doQuantile, double _minRatio, int _blockSize) {
  PredWeight(_predWeight, _predProb);
  Factory(_nTree, _nSamp, _smpReplace, _doQuantile, _minRatio, _blockSize);
}

// 
//
int Train::Training(int minH, int *facWidth, int *totBagCount, int *totQLeafWidth, int totLevels) {
  int auxRvSize = 0;

  // All of these factories require values from Predictor.
  //
  DataOrd::Factory();
  Node::Factory(totLevels, minH, auxRvSize);
  DecTree::ForestTrain(nTree);

  // 'rowVec' is reused by sampling without replacement, and is a convenient
  // size marker for sampling.
  //  IntegerVector rowVec(seq_len(Predictor::nRow)-1);
  //  RNGScope scope;

  for (int tn = 0; tn < nTree; tn++) {
    int bagCount = Response::SampleRows(Predictor::nRow);
    double *auxRv = 0;
    if (auxRvSize > 0) {
      //      RNGScope treeScope;
      // TODO:  Move to Node method.
      auxRv = Util::RUnif(auxRvSize);
    }
    int levels = Node::Levels(nSamp, bagCount, nPred, auxRv);
    int treeSize = PreTree::Produce(levels);
    DecTree::ConsumePretree(DataOrd::inBag, bagCount, treeSize, tn);
    // TODO:  Account for AccumHandler::handler->ClearTree();
  }
  int forestHeight = DecTree::AllTrees(facWidth, totBagCount, totQLeafWidth);

  DeFactory();
  DataOrd::DeFactory();
  SplitSig::DeFactory();
  Node::DeFactory();
  Predictor::DeFactory(); // Dispenses with training clone of 'x'.

  return forestHeight;
}

void Train::SampleWeights(double sWeight[]) {
  int nRow = Predictor::nRow;
  cdfOff = new int[nRow];
  sCDF = new double[nRow + 1];

  double recipNRow = 1.0 / nRow;
  double *placeVal = new double[nRow];
  for (int i = 0; i < nRow; i++)
    placeVal[i] = i * recipNRow;

  double sum = 0.0;
  for (int i = 0; i < nRow; i++) {
    double weight = sWeight[i];
    if (weight > 0.0) {
      sum += weight;
    }
  }
  double recip = 1.0 / sum;

  double cdf = 0.0;
  int j = 0;
  for (int i = 0; i < nRow; i++) {
    double weight = sWeight[i];
    if (weight > 0.0) {
      sCDF[i] = cdf;
      cdf += weight * recip;
      while (j < nRow && cdf > placeVal[j]) {
	cdfOff[j] = i;
	j++;
      }
    }
    else
      sCDF[i] = 0.0;
  }
  sCDF[nRow] = 1.0;

  //  for (int i = 0; i < nRow; i++) {
  //j = SampleDraw(sWeight[i]);
  //}
  delete [] placeVal;
}

// Moves up from the CDF next lowest to draw's fraction.
//
int Train::SampleDraw(double draw) {
  int j = draw * (Predictor::nRow - 1);
  //  cout << "draw:  " << draw << "  j:  " << j  << endl;
  int idx = cdfOff[j];
  while (idx <= Predictor::nRow && sCDF[idx] < draw) {
    idx++;
  }

  // ASSERTION:
  if (sCDF[idx-1] > draw || sCDF[idx] < draw)
    cout << draw << ":  choosing " << j << ":  " << sCDF[j] << " to " << sCDF[j+1] << endl;

  return idx - 1;
}

void Train::Quantiles(double *_qVec, int _qCells) {
  qCells = _qCells;
  qVec = _qVec;
}

void Train::WriteForest(int *rPreds, double *rSplits, double * rScores, int *rBumpL, int *rBumpR, int *rOrigins, int *rFacOff, int * rFacSplits) {
  DecTree::WriteForest(rPreds, rSplits, rScores, rBumpL, rBumpR, rOrigins, rFacOff, rFacSplits);

  // Dispenses with second load of predictor data (BlockData()).  Only client this late
  // appears to be use of 'nPredFac' to indicate presence of factor predictors.  Substitution
  // with an alternate indicator could allow this deallocation to be hoisted to the
  // Finish method for prediction.
  //
  Predictor::DeFactory();
}


void Train::WriteQuantile(double rQYRanked[], int rQRankOrigin[], int rQRank[], int rQRankCount[], int rQLeafPos[], int rQLeafExtent[]) {
  DecTree::WriteQuantile(rQYRanked, rQRankOrigin, rQRank, rQRankCount, rQLeafPos, rQLeafExtent);
}

// Ensures that the probability vector is large enough to reference
// 'liveCount' * 'nPred' predictors.  If not, reallocates.
//
void Train::LevelReset(int liveCount) {
    if (ruCt + nPred * liveCount > probSize) {   // Not enough slots left for this level.
      probSize <<= 1;
      treePredProb = Util::Sample(probSize);
      ruCt = 0;
      probResize++;
    }
    ruCt += nPred * liveCount;
}
