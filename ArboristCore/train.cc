// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "dectree.h"
#include "sample.h"
#include "train.h"
#include "predictor.h"
#include "dectree.h"
#include "index.h"
#include "response.h"
#include "splitsig.h"
#include "callback.h"
#include "math.h"

#include <iostream>
using namespace std;


int Train::accumRealloc = -1;
int Train::probResize = -1;
int Train::nTree = -1;
int Train::levelMax = -1;
int Train::qCells = -1;
double *Train::qVec = 0;
bool Train::doQuantiles = false;
double Train::minRatio = 0.0;
int Train::blockSize = -1;
int *Train::cdfOff = 0;
double *Train::sCDF = 0;

// Singleton factory:  everything is static.
//
void Train::Factory(int _nTree, bool _quantiles, double _minRatio, int _blockSize) {
  nTree = _nTree;
  doQuantiles = _quantiles;
  blockSize = _blockSize;
  minRatio = _minRatio;
  accumRealloc = 0;
  probResize = 0;

  // Employs heuristics to determine suitable vector sizes for level-based
  // operations.
  //
  unsigned twoL = 1; // 2^ #(levels-1)
  unsigned uN = Sample::NSamp();
  int balancedDepth = 1;
  while (twoL <= uN) { // Next power of two greater than 'nSamp'.
    balancedDepth++;
    twoL <<= 1;
  }

  // There could be as many as (bagCount - 1)/2 levels, in the case of a completely
  // left- or right-leaning tree.
  //
  // Two greater than balanced tree height is empirically well-suited to regression
  // trees.  Categorical trees may require "unlimited" depth.
  //
  levelMax = 1 << (accumExp >= (balancedDepth - 5) ? accumExp : balancedDepth - 5);

  // Initial estimate.  Must be wide enough to be visited by every split/predictor
  // combination at every level, so reallocation check is done at the end of every
  // level.
  //
  //  probSize = levelMax * (balancedDepth + 1) * Predictor::NPred();
}

// Employs the old reallocation heuristic of doubling the high watermark.
// This also happens to be safe, as there cannot be more than twice as
// many splits in the next level.
//
// N.B.:  Assumes trees trained sequentially, so that newer 'levelMax' values
// can be employed by later trees.  If trees are trained in parallel, then
// a guard must be employed to prevent unnecessary reallocation.
//
int Train::reLevel = 0;

int Train::ReFactory() {
  reLevel++;
  levelMax <<= 1;

  return levelMax;
}

void Train::DeFactory() {
  //  cout << probResize << " prob resizes, " << accumReFactory << " accum reallocations" << endl;
  if (cdfOff != 0) {
    delete [] cdfOff;
    delete [] sCDF;
    cdfOff = 0;
    sCDF = 0;
  }
  reLevel = 0;
}


void Train::IntBlock(int xBlock[], int _nrow, int _ncol) {
  Predictor::IntegerBlock(xBlock, _nrow, _ncol);
}

void Train::ResponseReg(double y[]) {
  Response::FactoryReg(y, levelMax);
}

int Train::ResponseCtg(const int y[], double yPerturb[]) {
  return Response::FactoryCtg(y, yPerturb, levelMax);
}

// 
//
int Train::Training(int minH, int *facWidth, int *totBagCount, int *totQLeafWidth, int totLevels) {
  SplitSig::Factory(levelMax, Predictor::NPred());
  IndexNode::Factory(minH, totLevels);
  DecTree::FactoryTrain(nTree, Predictor::NRow(), Predictor::NPred(), Predictor::NPredNum(), Predictor::NPredFac());
  for (int tn = 0; tn < nTree; tn++) {
    int bagCount = Response::SampleRows(levelMax);
    int treeSize = IndexNode::Levels();
    DecTree::ConsumePretree(Sample::inBag, bagCount, treeSize, tn);
    Response::TreeClearSt();
  }
  int forestHeight = DecTree::AllTrees(facWidth, totBagCount, totQLeafWidth);

  DeFactory();
  Sample::DeFactory();
  SplitSig::DeFactory();
  IndexNode::DeFactory();
  Predictor::DeFactory(); // Dispenses with training clone of 'x'.

  return forestHeight;
}

void Train::Quantiles(double *_qVec, int _qCells) {
  qCells = _qCells;
  qVec = _qVec;
}

void Train::WriteForest(int *rPreds, double *rSplits, double * rScores, int *rBump, int *rOrigins, int *rFacOff, int * rFacSplits) {
  DecTree::WriteForest(rPreds, rSplits, rScores, rBump, rOrigins, rFacOff, rFacSplits);

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
