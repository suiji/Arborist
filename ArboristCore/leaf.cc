// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.cc

   @brief Methods to train leaf components for an entire forest.

   @author Mark Seligman
 */

#include "leaf.h"
#include "sample.h"
#include "bv.h"
#include "predict.h"

#include <algorithm>

bool LeafTrain::thinLeaves = false;

void LeafTrain::Immutables(bool _thinLeaves) {
  thinLeaves = _thinLeaves;
}


void LeafTrain::DeImmutables() {
  thinLeaves = false;
}


/**
   @brief Training constructor.
 */
LeafTrain::LeafTrain(unsigned int nTree) :
  origin(vector<unsigned int>(nTree)),
  leafNode(vector<LeafNode>(0)),
  bagLeaf(vector<BagLeaf>(0)) {
}


LeafTrain::~LeafTrain() {
}


/**
 */
LeafTrainReg::LeafTrainReg(unsigned int nTree) :
  LeafTrain(nTree) {
}



LeafTrainReg::~LeafTrainReg() {
}


/**
   @brief Reserves leafNode space based on estimate.

   @return void.
 */
void LeafTrain::Reserve(unsigned int leafEst, unsigned int bagEst) {
  leafNode.reserve(leafEst);
  bagLeaf.reserve(bagEst);
}


/**
   @brief Reserves space based on leaf- and bag-count estimates.

   @return void.
 */
void LeafTrainReg::Reserve(unsigned int leafEst, unsigned int bagEst) {
  LeafTrain::Reserve(leafEst, bagEst);
}


/**
 */
void LeafTrainCtg::Reserve(unsigned int leafEst, unsigned int bagEst) {
  LeafTrain::Reserve(leafEst, bagEst);
  weight.reserve(leafEst * nCtg);
}


/**
   @brief Constructor for crescent forest.
 */
LeafTrainCtg::LeafTrainCtg(unsigned int nTree,
                 unsigned int rowTrain_,
                 unsigned int nCtg_)
  : LeafTrain(nTree),
    weight(vector<double>(0)),
    nCtg(nCtg_),
    weightScale(1.0 / (rowTrain_ * nTree))
{
}


LeafTrainCtg::~LeafTrainCtg() {
}


/**
   @brief Fills in leaves for a tree using current Sample.

   @param leafMap maps sampled indices to leaf indices.

   @param tIdx is the absolute tree index.

   @return void, with side-effected Leaf object.
 */
void LeafTrainReg::Leaves(const Sample *sample, const vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *max_element(leafMap.begin(), leafMap.end());
  NodeExtent(sample, leafMap, leafCount, tIdx);
  BagTree(sample, leafMap);//, tIdx);
  Scores(sample, leafMap, leafCount, tIdx);
}


/**
   @brief Records row, multiplicity and leaf index for bagged samples
   within a tree.
   For this scheme to work, samples indices must reference consecutive 
   bagged rows, as they currently do.

   @param leafMap maps sample indices to leaves.

   @param tIdx is the index of the current tree.

   @return void.
*/
void LeafTrain::BagTree(const Sample *sample, const vector<unsigned int> &leafMap) {
  if (thinLeaves)
    return;
  for (unsigned int sIdx = 0; sIdx < sample->getBagCount(); sIdx++) {
    bagLeaf.emplace_back(BagLeaf(leafMap[sIdx], sample->getSCount(sIdx)));
  }
}


/**
   @brief Derives scores for regression tree:  intialize, accumulate, divide.

   @param leafMap maps sample id to leaf index.

   @param leafCount is the number of leaves in the tree.

   @return void, with output parameter vector.
*/
void LeafTrainReg::Scores(const Sample *sample, const vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx) {
  vector<unsigned int> sCount(leafCount); // Per-leaf sample counts.
  fill(sCount.begin(), sCount.end(), 0);
  for (unsigned int sIdx = 0; sIdx < sample->getBagCount(); sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    ScoreAccum(tIdx, leafIdx, sample->getSum(sIdx));
    sCount[leafIdx] += sample->getSCount(sIdx);
  }

  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    ScoreScale(tIdx, leafIdx, sCount[leafIdx]);
  }
}


/**
   @brief Writes the current tree origin and computes the extent of each leaf node.

   @param leafCount is the number of leaves in the current tree.

   @void, with count-adjusted leaf nodes.
 */
void LeafTrain::NodeExtent(const Sample *sample, vector<unsigned int> leafMap, unsigned int leafCount, unsigned int tIdx) {
  unsigned int leafBase = leafNode.size();
  origin[tIdx] = leafBase;

  LeafNode init;
  init.Init();
  leafNode.insert(leafNode.end(), leafCount, init);
  for (unsigned int sIdx = 0; sIdx < sample->getBagCount(); sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    leafNode[leafBase + leafIdx].Count()++;
  }
}


/**
   @brief Computes leaf weights and scores for a classification tree.

   @return void, with side-effected weights and forest terminals.
 */
void LeafTrainCtg::Leaves(const Sample *sample, const vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *max_element(leafMap.begin(), leafMap.end());
  NodeExtent(sample, leafMap, leafCount, tIdx);
  BagTree(sample, leafMap);//, tIdx);
  Scores((SampleCtg*) sample, leafMap, leafCount, tIdx);
}


/**
   @brief Weights and scores the leaves for a classification tree.

   @param sampleCtg is the sampling vector for the current tree.
   
   @param leafMap maps sample indices to leaf indices.

   @param treeOrigin is the base leaf index of the current tree.

   @return void, with side-effected weight vector.
 */
void LeafTrainCtg::Scores(const SampleCtg *sample,
                          const vector<unsigned int> &leafMap,
                          unsigned int leafCount,
                          unsigned int tIdx) {
  WeightInit(leafCount);

  vector<double> leafSum(leafCount);
  fill(leafSum.begin(), leafSum.end(), 0.0);
  for (unsigned int sIdx = 0; sIdx < sample->getBagCount(); sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    FltVal sum;
    unsigned int ctg;
    sample->RefLeaf(sIdx, sum, ctg);
    leafSum[leafIdx] += sum;
    WeightAccum(tIdx, leafIdx, ctg, sum);
  }

  // Scales weights by leaf for probabilities.
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    double maxWeight = 0.0;
    unsigned int argMax = 0;
    double recipSum = 1.0 / leafSum[leafIdx];
    for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
      double thisWeight = WeightScale(tIdx, leafIdx, ctg, recipSum);
      if (thisWeight > maxWeight) {
        maxWeight = thisWeight;
        argMax = ctg;
      }
    }
    ScoreSet(tIdx, leafIdx, argMax + maxWeight * weightScale);
  }
}


/**
 */
LeafReg::LeafReg(const unsigned int _origin[],
                 unsigned int _nTree,
                 const LeafNode _leafNode[],
                 unsigned int _leafCount,
                 const class BagLeaf _bagLeaf[],
                 unsigned int _bagLeafTot,
                 const double *_yTrain,
                 double _meanTrain,
                 unsigned int _rowPredict) :
  Leaf(_origin, _nTree, _leafNode, _leafCount, _bagLeaf, _bagLeafTot),
  yTrain(_yTrain),
  meanTrain(_meanTrain),
  offset(vector<unsigned int>(leafCount)),
  defaultScore(MeanTrain()),
  yPred(vector<double>(_rowPredict)) {
  Offsets();
}


/**
   @brief Constructor for trained forest:  vector lengths final.
 */
LeafCtg::LeafCtg(const unsigned int _origin[],
                         unsigned int _nTree,
                         const class LeafNode _leafNode[],
                         unsigned int _leafCount,
                         const class BagLeaf _bagLeaf[],
                         unsigned int _bagLeafTot,
                         const double _weight[],
                 unsigned int _ctgTrain,
                 unsigned int _rowPredict,
                 bool doProb) :
  Leaf(_origin,
       _nTree,
       _leafNode,
       _leafCount,
       _bagLeaf,
       _bagLeafTot),

  weight(_weight),
  ctgTrain(_ctgTrain),
  yPred(vector<unsigned int>(_rowPredict)),
  // Can only predict trained categories, so census and
  // probability matrices have 'ctgTrain' columns.
  defaultScore(ctgTrain),
  defaultWeight(vector<double>(ctgTrain)),
  votes(vector<double>(_rowPredict * ctgTrain)),
  census(vector<unsigned int>(_rowPredict * ctgTrain)),
  prob(vector<double>(doProb ? _rowPredict * ctgTrain : 0)) {
  fill(defaultWeight.begin(), defaultWeight.end(), -1.0);
  fill(votes.begin(), votes.end(), 0.0);
}


/**
   @brief Prediction constructor.
 */
Leaf::Leaf(const unsigned int *_origin,
                   unsigned int _nTree,
                   const LeafNode *_leafNode,
                   unsigned int _leafCount,
                   const BagLeaf *_bagLeaf,
           unsigned int _bagTot) :
  origin(_origin),
  leafNode(_leafNode),
  bagLeaf(_bagLeaf),
  nTree(_nTree),
  leafCount(_leafCount),
  bagLeafTot(_bagTot) {
}


Leaf::~Leaf() {
}


/**
   @brief Accumulates exclusive sum of counts for offset lookup.  Only
   client is quantile regression:  exits if bagLeaf[] empty.

   @return void, with side-effected reference vector.
 */
void LeafReg::Offsets() {
  if (bagLeafTot == 0)
    return;
  unsigned int countAccum = 0;
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    offset[leafIdx] = countAccum;
    countAccum += Extent(leafIdx);
  }
  // Post-condition:  countAccum == bagTot
}



/**
   @brief Assigns a forest-wide default weighting value to each category.

   @return void, with output reference parameter.
 */
void LeafCtg::DefaultWeight(vector<double> &defaultWeight) const {
  unsigned int idx = 0;
  for (unsigned int forestIdx = 0; forestIdx < leafCount; forestIdx++) {
    for (unsigned int ctg = 0; ctg < ctgTrain; ctg++) {
      defaultWeight[ctg] += weight[idx++];
    }
  }
  for (unsigned int ctg = 0; ctg < ctgTrain; ctg++) {
    defaultWeight[ctg] /= leafCount;
  }
}


/**
 */
void LeafReg::populate(const BitMatrix *baggedRows,
                         vector<vector<unsigned int> > &rowTree,
                         vector<vector<unsigned int> > &sCountTree,
                         vector<vector<double> > &scoreTree,
                         vector<vector<unsigned int> >&extentTree) const {
  Leaf::populate(baggedRows, rowTree, sCountTree);
  nodeExport(scoreTree, extentTree);
}


size_t LeafTrain::NodeBytes() const {
  return leafNode.size() * sizeof(LeafNode);
}


size_t LeafTrain::BLBytes() const {
  return bagLeaf.size() * sizeof(BagLeaf);
}


/** 
    @brief Serializes the internally-typed objects, 'LeafNode', as well
    as the unsigned integer (packed bit) vector, "bagBits".
*/
void LeafTrain::Serialize(unsigned char *leafRaw, unsigned char *blRaw) const {
  for (size_t i = 0; i < NodeBytes(); i++) {
    leafRaw[i] = ((unsigned char*) &leafNode[0])[i];
  }

  for (size_t i = 0; i < BLBytes(); i++) {
    blRaw[i] = ((unsigned char*) &bagLeaf[0])[i];
  }
}


/**
   @brief Static exporter of BagLeaf vector into per-tree vector of vectors.

   @return void, with output reference parameters.
 */
void Leaf::populate(const BitMatrix *baggedRows,
                  vector< vector<unsigned int> > &rowTree,
                  vector< vector<unsigned int> > &sCountTree) const {
  unsigned int leafOff = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int row = 0; row < baggedRows->getStride(); row++) {
      if (baggedRows->testBit(tIdx, row)) {
        rowTree[tIdx].emplace_back(row);
        sCountTree[tIdx].emplace_back(bagLeaf[leafOff++].getSCount());
      }
    }
  }
}


/**
   @brief Exports LeafNode into vectors of per-tree vectors.



   @return void, with output reference parameters.
 */
void Leaf::nodeExport(vector<vector<double> > &score,
                          vector<vector<unsigned int> > &extent) const {
  unsigned int forestIdx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int leafIdx = 0; leafIdx < TreeHeight(tIdx); leafIdx++) {
      score[tIdx].push_back(leafNode[forestIdx].Score());
      extent[tIdx].push_back(leafNode[forestIdx].Extent());
      forestIdx++;
    }
  }
}


/**
  @brief Sets regression scores from leaf predictions.

  @return void, with output refererence vector.
 */
void LeafReg::ScoreBlock(const Predict *predict,
                         unsigned int rowStart,
                         unsigned int rowEnd) {
  OMPBound blockRow;
  OMPBound blockSup = (OMPBound) (rowEnd - rowStart);

#pragma omp parallel default(shared) private(blockRow)
  {
#pragma omp for schedule(dynamic, 1)
  for (blockRow = 0; blockRow < blockSup; blockRow++) {
      double score = 0.0;
      int treesSeen = 0;
      for (unsigned int tc = 0; tc < nTree; tc++) {
        unsigned int termIdx;
        if (!predict->isBagged(blockRow, tc, termIdx)) {
          treesSeen++;
          score += getScore(tc, termIdx);
        }
      }
      yPred[rowStart + blockRow] = treesSeen > 0 ? score / treesSeen : defaultScore;
    }
  }
}


/**
   @brief Computes score from leaf predictions.

   @return internal vote table, with output reference vector.
 */
void LeafCtg::ScoreBlock(const Predict *predict,
                         unsigned int rowStart,
                         unsigned int rowEnd) {
  OMPBound blockRow;
  OMPBound blockSup = (OMPBound) (rowEnd - rowStart);
// TODO:  Recast loop by blocks, to avoid
// false sharing.
#pragma omp parallel default(shared) private(blockRow)
  {
#pragma omp for schedule(dynamic, 1)
  for (blockRow = 0; blockRow < blockSup; blockRow++) {
    double *prediction = &votes[getTrainIdx(rowStart + blockRow, 0)];
    unsigned int treesSeen = 0;
    for (unsigned int tc = 0; tc < nTree; tc++) {
      unsigned int termIdx;
      if (!predict->isBagged(blockRow, tc, termIdx)) {
        treesSeen++;
        double val = getScore(tc, termIdx);
        unsigned int ctg = val; // Truncates jittered score for indexing.
        prediction[ctg] += 1 + val - ctg;
      }
    }
    if (treesSeen == 0) {
      for (unsigned int ctg = 0; ctg < ctgTrain; ctg++) {
        prediction[ctg] = 0.0;
      }
      prediction[DefaultScore()] = 1;
    }
  }
  }
  if (prob.size() != 0) {
    ProbBlock(predict, rowStart, rowEnd);
  }
}


/**
    Fills in proability matrix:  rowPredict x ctgTrain.
 */
void LeafCtg::ProbBlock(const Predict *predict,
                        unsigned int rowStart,
                        unsigned int rowEnd) {
  for (unsigned int blockRow = 0; blockRow < rowEnd - rowStart; blockRow++) {
    double *probRow = &prob[getTrainIdx(rowStart + blockRow, 0)];
    double rowSum = 0.0;
    unsigned int treesSeen = 0;
    for (unsigned int tc = 0; tc < nTree; tc++) {
      unsigned int termIdx;
      if (!predict->isBagged(blockRow, tc, termIdx)) {
        treesSeen++;
        for (unsigned int ctg = 0; ctg < ctgTrain; ctg++) {
          double idxWeight = WeightCtg(tc, termIdx, ctg);
          probRow[ctg] += idxWeight;
          rowSum += idxWeight;
        }
      }
    }
    if (treesSeen == 0) {
      rowSum = DefaultWeight(probRow);
    }

    double scale = 1.0 / rowSum;
    for (unsigned int ctg = 0; ctg < ctgTrain; ctg++)
      probRow[ctg] *= scale;
  }
}


/**
   @brief Voting for non-bagged prediction.  Rounds jittered scores to category.

   @param yCtg outputs predicted response.

   @return void, with output reference vector.
*/
void LeafCtg::Vote() {
  OMPBound rowSup = yPred.size();//rowPredict;
  OMPBound row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < rowSup; row++) {
    unsigned int argMax = ctgTrain;
    double scoreMax = 0.0;
    double *scoreRow = &votes[getTrainIdx(row,0)];
    for (unsigned int ctg = 0; ctg < ctgTrain; ctg++) {
      double ctgScore = scoreRow[ctg]; // Jittered vote count.
      if (ctgScore > scoreMax) {
        scoreMax = ctgScore;
        argMax = ctg;
      }
      census[getTrainIdx(row, ctg)] = ctgScore; // De-jittered.
    }
    yPred[row] = argMax;
  }
  }
}


/**
 */
void LeafCtg::populate(const BitMatrix *baggedRows,
                     vector<vector<unsigned int> > &rowTree,
                     vector<vector<unsigned int> > &sCountTree,
                     vector<vector<double> > &scoreTree,
                     vector<vector<unsigned int> > &extentTree,
                     vector<vector<double> > &weightTree) const {
  Leaf::populate(baggedRows, rowTree, sCountTree);
  nodeExport(scoreTree, extentTree);

  unsigned int off = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int leafIdx = 0; leafIdx < TreeHeight(tIdx); leafIdx++) {
      for (unsigned int ctg = 0; ctg < ctgTrain; ctg++) {
        weightTree[tIdx].push_back(weight[off++]);
      }
    }
  }
}

/**
   @brief Lazily sets default score.

   @return default score.
 */
unsigned int LeafCtg::DefaultScore() {
  if (defaultScore >= ctgTrain) {
    DefaultInit();

    defaultScore = 0;
    double weightMax = defaultWeight[0];
    for (unsigned int ctg = 1; ctg < ctgTrain; ctg++) {
      if (defaultWeight[ctg] > weightMax) {
        defaultScore = ctg;
        weightMax = defaultWeight[ctg];
      }
    }
  }

  return defaultScore;
}


/**
   @brief Lazily sets default weight.
   TODO:  Ensure error if called when no bag present.

   @return void.
 */
void LeafCtg::DefaultInit() {
  if (defaultWeight[0] < 0.0) { // Unseen.
    fill(defaultWeight.begin(), defaultWeight.end(), 0.0);
    DefaultWeight(defaultWeight);
  }
}


double LeafCtg::DefaultWeight(double *weightPredict) {
  double rowSum = 0.0;
  for (unsigned int ctg = 0; ctg < ctgTrain; ctg++) {
    weightPredict[ctg] = defaultWeight[ctg];
    rowSum += weightPredict[ctg];
  }
  return rowSum;
}
