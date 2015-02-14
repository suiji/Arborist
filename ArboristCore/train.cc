// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file train.cc

   @brief Main entry from front end for training.

   @author Mark Seligman
 */

#include "dectree.h"
#include "quant.h"
#include "sample.h"
#include "train.h"
#include "predictor.h"
#include "dectree.h"
#include "index.h"
#include "response.h"
#include "splitsig.h"
#include "callback.h"
#include "math.h"

// Testing only:
//#include <iostream>
using namespace std;


int Train::accumRealloc = -1;
int Train::probResize = -1;
int Train::nTree = -1;
int Train::levelMax = -1;
double Train::minRatio = 0.0;
int Train::blockSize = -1;
int *Train::cdfOff = 0;
double *Train::sCDF = 0;
int Train::reLevel = 0;

/**
   @brief Singleton factory:  everything is static.

   @param _nTree is the requested number of trees.

   @param _minRatio is a threshold ratio for determining whether to split.

   @param _blockSize is a predictor-blocking heuristic for parallel implementations.

   @return level-max value.
*/
int Train::Factory(int _nTree, double _minRatio, int _blockSize) {
  nTree = _nTree;
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

  return levelMax;
}


/**
   @brief Determines next level-max value.

   @return next level-max value.
   Employs the old reallocation heuristic of doubling the high watermark.
   This also happens to be safe, as there cannot be more than twice as
   many splits in the next level.

   N.B.:  Assumes trees trained sequentially, so that newer 'levelMax' values
   can be employed by later trees.  If trees are trained in parallel, then
   a guard must be employed to prevent unnecessary reallocation.
 */
int Train::ReFactory() {
  reLevel++;
  levelMax <<= 1;

  return levelMax;
}

/**
   @brief Finalizer.
*/
void Train::DeFactory() {
  if (cdfOff != 0) {
    delete [] cdfOff;
    delete [] sCDF;
    cdfOff = 0;
    sCDF = 0;
  }
  reLevel = 0;
}

/**
   @brief Main entry for training after singleton factory.

   @param minH is the minimal index node size on which to split.

   @param _quantiles is true iff quantiles have been requested.

   @param facWidth outputs the sum of factor cardinalities.

   @param totBagCount outputs the sum of all tree in-bag sizes.

   @param totLevels, if positive, limits the number of levels to build.

   @return void.
*/
int Train::Training(int minH, bool _quantiles, int totLevels, int &facWidth, int &totBagCount) {
  totBagCount = 0;
  SplitSig::Factory(levelMax, Predictor::NPred());
  IndexNode::Factory(minH, totLevels);
  DecTree::FactoryTrain(nTree, Predictor::NRow(), Predictor::NPred(), Predictor::NPredNum(), Predictor::NPredFac());
  Quant::FactoryTrain(Predictor::NRow(), nTree, _quantiles);
  for (int tn = 0; tn < nTree; tn++) {
    int bagCount = Response::SampleRows(levelMax);
    int treeSize = IndexNode::Levels();
    DecTree::ConsumePretree(Sample::inBag, bagCount, treeSize, tn);
    Response::TreeClearSt();
    totBagCount += bagCount;
  }
  int forestHeight = DecTree::ConsumeTrees(facWidth);

  DeFactory();
  Sample::DeFactory();
  SplitSig::DeFactory();
  IndexNode::DeFactory();
  Predictor::DeFactory(); // Dispenses with training clone of 'x'.

  return forestHeight;
}

/**
   @brief Writes decision forest to storage provided by front end.

   @rPreds outputs splitting predictors.

   @rSplits outputs splitting values.

   @rScores outputs leaf scores.

   @rBump outputs branch increments.

   @rOrigins outputs offsets of individual trees.

   @rFacOff outputs offsets of spitting bit vectors.

   @rFacSplits outputs factor splitting bit vectors.

   @return void, with output parameter vectors.
 */
void Train::WriteForest(int *rPreds, double *rSplits, double * rScores, int *rBump, int *rOrigins, int *rFacOff, int * rFacSplits) {
  DecTree::WriteForest(rPreds, rSplits, rScores, rBump, rOrigins, rFacOff, rFacSplits);

  // Dispenses with second load of predictor data (BlockData()).  Only client this late
  // appears to be use of 'nPredFac' to indicate presence of factor predictors.  Substitution
  // with an alternate indicator could allow this deallocation to be hoisted to the
  // Finish method for prediction.
  //
  Predictor::DeFactory();
}

/**
   @brief Writes quantile information to storage provided by front end.

   @rQYRanked outputs the ranked response values.

   @rQRankOrigins outputs the leaf offsets for the start of each tree.

   @rQRank outputs the quantile ranks.

   @rQRankCount outputs the count of quantile ranks.

   @rQLeafPos outputs the quantile leaf positions.

   @rQLeafExtent outputs the quantile leaf sizes.

   @return void, with output parameter vectors.
*/
void Train::WriteQuantile(double rQYRanked[], int rQRankOrigin[], int rQRank[], int rQRankCount[], int rQLeafPos[], int rQLeafExtent[]) {
  Quant::Write(rQYRanked, rQRankOrigin, rQRank, rQRankCount, rQLeafPos, rQLeafExtent);
}
