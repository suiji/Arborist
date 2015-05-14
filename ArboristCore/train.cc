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
#include "pretree.h"
#include "splitsig.h"
#include "callback.h"


// Testing only:
//#include <iostream>
using namespace std;

int Train::accumRealloc = -1;
int Train::probResize = -1;
int Train::nTree = -1;
int Train::treeBlock = -1;
int *Train::cdfOff = 0;
double *Train::sCDF = 0;


/**
   @brief Singleton factory:  everything is static.

   @param _nTree is the requested number of trees.

   @param _treeBlock is the number of PreTree objects to brace for MPI-style parallelism.

   @return void.
*/
void Train::Factory(int _nTree, int _treeBlock) {
  nTree = _nTree;
  treeBlock = _treeBlock;
  accumRealloc = 0;
  probResize = 0;
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
}

/**
   @brief Main entry for training after singleton factory.

   @param minH is the minimal index node size on which to split.

   @param quantiles is true iff quantiles have been requested.

   @param minRatio is a threshold ratio for determining whether to split.

   @param facWidth outputs the sum of factor cardinalities.

   @param totBagCount outputs the sum of all tree in-bag sizes.

   @param totLevels, if positive, limits the number of levels to build.

   @return void.
*/
int Train::Training(int minH, bool _quantiles, double minRatio, int totLevels, int &facWidth, int &totBagCount) {
  SplitSig::Immutables(Predictor::NPred(), minRatio);
  Index::Immutables(minH, totLevels, Sample::NSamp());
  DecTree::FactoryTrain(nTree, Predictor::NRow(), Predictor::NPred(), Predictor::NPredNum(), Predictor::NPredFac());
  Quant::FactoryTrain(Predictor::NRow(), nTree, _quantiles);
  PreTree::Immutables(Predictor::NRow(), Sample::NSamp(), minH);

  PredOrd *predOrd = Predictor::Order();
  totBagCount = TrainForest(predOrd, nTree);
  delete [] predOrd;
  
  int forestHeight = DecTree::ConsumeTrees(facWidth);

  DeFactory();
  Sample::DeImmutables();
  SplitSig::DeImmutables();
  Index::DeImmutables();
  PreTree::DeImmutables();
  Predictor::DeFactory(); // Dispenses with training clone of 'x'.

  return forestHeight;
}

/**
  @brief Trains the requisite number of trees.

  @param treeCount is the number of trees to construct.

  @return Sum of bag counts.
*/
int Train::TrainForest(const PredOrd *predOrd, int treeCount) {
  int totBagCount = 0;
  int tn;

  totBagCount = TrainZero(predOrd);
  
  for (tn = 1; tn < treeCount - treeBlock; tn += treeBlock) {
    totBagCount += TrainBlock(predOrd, tn, treeBlock);
  }
  if (tn < treeCount)
    totBagCount += TrainBlock(predOrd, tn, treeCount - tn);
  
  return totBagCount;
}


/**
   @brief Trains tree zero separately and records height information.

   @param predOrd is the ordered predictor set.

   @return void.
 */
int Train::TrainZero(const PredOrd *predOrd) {
  PreTree **ptBlock = Index::BlockTrees(predOrd, 1);
  PreTree::RefineHeight(ptBlock[0]->TreeHeight());

  return DecTree::BlockConsume(ptBlock, 1, 0);
}

/**
   @brief Trains a block of pretrees, then builds decision trees from them.  Training in blocks facilitates coarse-grain parallel treatments, such as map/reduce or MPI.

   @param predOrd is the sorted predictor table.

   @param tn is the index of the first tree in the current block.

   @param treeBlock is the number of trees in the block.

   @return sum of bag counts of trees built.
 */
int Train::TrainBlock(const PredOrd *predOrd, int tn, int treeBlock) {
  PreTree **ptBlock = Index::BlockTrees(predOrd, treeBlock);

  return DecTree::BlockConsume(ptBlock, treeBlock, tn);
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
