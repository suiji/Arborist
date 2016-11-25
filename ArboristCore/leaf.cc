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
#include "predblock.h"
#include "sample.h"
#include "bv.h"

#include <algorithm>

//#include <iostream>
//using namespace std;

bool Leaf::thinLeaves = false;

void Leaf::Immutables(bool _thinLeaves) {
  thinLeaves = _thinLeaves;
}


void Leaf::DeImmutables() {
  thinLeaves = false;
}


/**
   @brief Prediction constructor.
 */
Leaf::Leaf(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits) : origin(_origin), nTree(origin.size()), leafNode(_leafNode), bagLeaf(_bagLeaf), bagRow(new BitMatrix(_bagBits, _bagBits.size() / nTree, nTree)), offset(std::vector<unsigned int>(leafNode.size())) {
  ForestOffsets();
}


/**
   @breif Training constructor.
 */
Leaf::Leaf(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, unsigned int rowTrain) : origin(_origin), nTree(origin.size()), leafNode(_leafNode), bagLeaf(_bagLeaf), bagRow(new BitMatrix(_bagBits, rowTrain, nTree)), offset(std::vector<unsigned int>(0)) {
}


/**
   @brief Accumulates exclusive sum of counts for offset lookup.

   @return void, with side-effected reference vector.
 */
void Leaf::ForestOffsets() {
  unsigned int countAccum = 0;
  for (unsigned int leafIdx = 0; leafIdx < offset.size(); leafIdx++) {
    offset[leafIdx] = countAccum;
    countAccum += Extent(leafIdx);
  }
  // Post-condition:  countAccum == bagCount
}


Leaf::~Leaf() {
  delete bagRow;
}


/**
 */
LeafReg::LeafReg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits) : Leaf(_origin, _leafNode, _bagLeaf, _bagBits) {
}

/**
 */
LeafReg::LeafReg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, unsigned int rowTrain) : Leaf(_origin, _leafNode, _bagLeaf, _bagBits, rowTrain) {
}


LeafReg::~LeafReg() {
}


/**
   @brief Reserves leafNode space based on estimate.

   @return void.
 */
void Leaf::Reserve(unsigned int leafEst, unsigned int bagEst) {
  leafNode.reserve(leafEst);
  bagLeaf.reserve(bagEst);
}


/**
   @brief Reserves space based on leaf- and bag-count estimates.

   @return void.
 */
void LeafReg::Reserve(unsigned int leafEst, unsigned int bagEst) {
  Leaf::Reserve(leafEst, bagEst);
}


/**
 */
void LeafCtg::Reserve(unsigned int leafEst, unsigned int bagEst) {
  Leaf::Reserve(leafEst, bagEst);
  weight.reserve(leafEst * ctgWidth);
}


/**
   @brief Constructor for crescent forest.
 */
LeafCtg::LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, unsigned int rowTrain, std::vector<double> &_weight, unsigned int _ctgWidth) : Leaf(_origin, _leafNode, _bagLeaf, _bagBits, rowTrain), weight(_weight), ctgWidth(_ctgWidth) {
}


/**
   @brief Constructor for trained forest:  vector lengths final.
 */
LeafCtg::LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, std::vector<double> &_weight) : Leaf(_origin, _leafNode, _bagLeaf, _bagBits), weight(_weight), ctgWidth(weight.size() / NodeCount()) {
}


LeafCtg::~LeafCtg() {
}


/**
   @brief Derives and copies regression leaf information.

   @param leafExtent gives leaf width at forest index.

   @return bag count, with output parameter vectors.
 */
void LeafReg::Leaves(const PMTrain *pmTrain, const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *std::max_element(leafMap.begin(), leafMap.end());
  NodeExtent(sample, leafMap, leafCount, tIdx);
  BagTree(sample, leafMap, tIdx);
  Scores(sample, leafMap, leafCount, tIdx);
}


/**
   @brief Derives scores for regression tree:  intialize, accumulate, divide.

   @param leafMap maps sample id to leaf index.

   @param leafCount is the number of leaves in the tree.

   @return void, with output parameter vector.
*/
void LeafReg::Scores(const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx) {
  std::vector<unsigned int> sCount(leafCount); // Per-leaf sample counts.
  std::fill(sCount.begin(), sCount.end(), 0);
  for (unsigned int sIdx = 0; sIdx < sample->BagCount(); sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    ScoreAccum(tIdx, leafIdx, sample->Sum(sIdx));
    sCount[leafIdx] += sample->SCount(sIdx);
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
void Leaf::NodeExtent(const Sample *sample, std::vector<unsigned int> leafMap, unsigned int leafCount, unsigned int tIdx) {
  unsigned int leafBase = leafNode.size();
  origin[tIdx] = leafBase;

  LeafNode init;
  init.Init();
  leafNode.insert(leafNode.end(), leafCount, init);
  for (unsigned int sIdx = 0; sIdx < sample->BagCount(); sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    leafNode[leafBase + leafIdx].Count()++;
  }
}


/**
   @brief Computes leaf weights and scores for a classification tree.

   @return void, with side-effected weights and forest terminals.
 */
void LeafCtg::Leaves(const PMTrain *pmTrain, const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *std::max_element(leafMap.begin(), leafMap.end());
  NodeExtent(sample, leafMap, leafCount, tIdx);
  BagTree(sample, leafMap, tIdx);
  Scores(pmTrain, (SampleCtg*) sample, leafMap, leafCount, tIdx);
}


/**
   @brief Weights and scores the leaves for a classification tree.

   @param sampleCtg is the sampling vector for the current tree.
   
   @param leafMap maps sample indices to leaf indices.

   @param treeOrigin is the base leaf index of the current tree.

   @return void, with side-effected weight vector.
 */
void LeafCtg::Scores(const PMTrain *pmTrain, const SampleCtg *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx) {
  WeightInit(leafCount);

  std::vector<double> leafSum(leafCount);
  std::fill(leafSum.begin(), leafSum.end(), 0.0);
  for (unsigned int sIdx = 0; sIdx < sample->BagCount(); sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    FltVal sum;
    unsigned int dummy;
    unsigned int ctg = sample->Ref(sIdx, sum, dummy);
    leafSum[leafIdx] += sum;
    WeightAccum(tIdx, leafIdx, ctg, sum);
  }

  // Scales weights by leaf for probabilities.
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    double maxWeight = 0.0;
    unsigned int argMax = 0;
    double recipSum = 1.0 / leafSum[leafIdx];
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      double thisWeight = WeightScale(tIdx, leafIdx, ctg, recipSum);
      if (thisWeight > maxWeight) {
	maxWeight = thisWeight;
        argMax = ctg;
      }
    }
    ScoreSet(tIdx, leafIdx, argMax + maxWeight / (pmTrain->NRow() * NTree()));
  }
}


/**
 */
unsigned int LeafCtg::LeafCount(std::vector<unsigned int> _origin, unsigned int weightLen, unsigned int _ctgWidth, unsigned int tIdx) {
  return LeafNode::LeafCount(_origin, weightLen / _ctgWidth, tIdx);
}


/**
   @brief Static entry for recomputing tree bag count.

   @param _origin is the index of offsets into the node vector.

   @param _leafNode is the forest-wide leaf set.

   @param tIdx is the index of a tree.

   @return bag count of tree indexed by 'tIdx'.
 */
unsigned int Leaf::BagCount(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, unsigned int tIdx) {
  unsigned int leafFirst = _origin[tIdx];
  unsigned int leafSup = tIdx < _origin.size() - 1 ? _origin[tIdx + 1] : _leafNode.size();
  unsigned int bagCount = 0;
  for (unsigned int leafIdx = leafFirst; leafIdx < leafSup; leafIdx++) {
    bagCount += _leafNode[leafIdx].Extent();
  }
  
  return bagCount;
}


/**
   @brief Assigns a forest-wide default weighting value to each category.

   @return void, with output reference parameter.
 */
void LeafCtg::ForestWeight(double *defaultWeight) const {
  unsigned int idx = 0;
  unsigned int forestLeaves = weight.size() / ctgWidth;
  for (unsigned int forestIdx = 0; forestIdx < forestLeaves; forestIdx++) {
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      defaultWeight[ctg] += weight[idx++];
    }
  }
  for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
    defaultWeight[ctg] /= forestLeaves;
  }
}


/**
 */
void LeafReg::Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, const std::vector<BagLeaf> &_bagLeaf, std::vector<std::vector<unsigned int> > &rowTree, std::vector<std::vector<unsigned int> > &sCountTree, std::vector<std::vector<double> > &scoreTree, std::vector<std::vector<unsigned int> >&extentTree) {
  Leaf::Export(_origin, _leafNode, _bagLeaf, rowTree, sCountTree);
  LeafNode::Export(_origin, _leafNode, scoreTree, extentTree);
  unsigned int bagOrig = 0;
  for (unsigned int tIdx = 0; tIdx < _origin.size(); tIdx++) {
    unsigned int bagCount = BagCount(_origin, _leafNode, tIdx);
    bagOrig += bagCount;
  }
}


/**
   @brief Static exporter of BagLeaf vector into per-tree vector of vectors.

   @return void, with output reference parameters.
 */
void Leaf::Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, const std::vector<BagLeaf> &_bagLeaf, std::vector< std::vector<unsigned int> > &rowTree, std::vector< std::vector<unsigned int> >&sCountTree) {
  unsigned int _nTree = _origin.size();
  unsigned int bagOrig = 0;
  for (unsigned int tIdx = 0; tIdx < _nTree; tIdx++) {
    unsigned int bagCount = BagCount(_origin, _leafNode, tIdx);
    rowTree[tIdx] = std::vector<unsigned int>(bagCount);
    sCountTree[tIdx] = std::vector<unsigned int>(bagCount);
    TreeExport(_bagLeaf, bagOrig, bagCount, rowTree[tIdx], sCountTree[tIdx]);
    bagOrig += bagCount;
  }
}


/**
   @brief Static exporter of LeafNode into per-tree vector of vectors.

   @return void, with output reference parameters.
 */
void LeafNode::Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, std::vector< std::vector<double> > &_score, std::vector< std::vector<unsigned int> > &_extent) {
  for (unsigned int tIdx = 0; tIdx < _origin.size(); tIdx++) {
    unsigned int leafCount = LeafCount(_origin, _leafNode.size(), tIdx);
    _score[tIdx] = std::vector<double>(leafCount);
    _extent[tIdx] = std::vector<unsigned int>(leafCount);
    TreeExport(_leafNode, _origin[tIdx], leafCount, _score[tIdx], _extent[tIdx]); 
  }
}


void Leaf::TreeExport(const std::vector<BagLeaf> &_bagLeaf, unsigned int bagOrig, unsigned int bagCount, std::vector<unsigned int> &rowTree, std::vector<unsigned int> &sCountTree) {
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    rowTree[sIdx] = 0; // FIX:  Obtain rows from BagRow bits, in order.
    sCountTree[sIdx] = _bagLeaf[bagOrig + sIdx].SCount();
  }
}


/**
   @brief Per-tree exporter into separate vectors.
 */
void LeafNode::TreeExport(const std::vector<LeafNode> &_leafNode, unsigned int treeOrig, unsigned int leafCount, std::vector<double> &scoreTree, std::vector<unsigned int> &extentTree) {
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    _leafNode[treeOrig + leafIdx].Ref(scoreTree[leafIdx], extentTree[leafIdx]);
  }
}




/**
 */
void LeafCtg::Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, const std::vector<BagLeaf> &_bagLeaf, const std::vector<double> &_weight, unsigned int _ctgWidth, std::vector<std::vector<unsigned int> > &rowTree, std::vector<std::vector<unsigned int> > &sCountTree, std::vector<std::vector<double> > &scoreTree, std::vector<std::vector<unsigned int> > &extentTree, std::vector<std::vector<double> > &weightTree) {
  Leaf::Export(_origin, _leafNode, _bagLeaf, rowTree, sCountTree);
  LeafNode::Export(_origin, _leafNode, scoreTree, extentTree);
  for (unsigned int tIdx = 0; tIdx < _origin.size(); tIdx++) {
    unsigned int leafCount = LeafCount(_origin, _weight.size(), _ctgWidth, tIdx);
    weightTree[tIdx] = std::vector<double>(leafCount * _ctgWidth);
    TreeExport(_weight, _ctgWidth, _origin[tIdx] * _ctgWidth, leafCount, weightTree[tIdx]);
  }
}


void LeafCtg::TreeExport(const std::vector<double> &leafWeight, unsigned int _ctgWidth, unsigned int treeOrig, unsigned int leafCount, std::vector<double> &weightTree) {
  unsigned int off = 0;
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    for (unsigned int ctg = 0; ctg < _ctgWidth; ctg++) {
      weightTree[off] = leafWeight[treeOrig + off];
      off++;
    }
  }
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
void Leaf::BagTree(const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) {
  std::vector<unsigned int> sample2Row(sample->BagCount());
  sample->RowInvert(sample2Row);
  for (unsigned int sIdx = 0; sIdx < sample->BagCount(); sIdx++) {
    bagRow->SetBit(sample2Row[sIdx], tIdx);
    if (!thinLeaves) {
      BagLeaf lb;
      lb.Init(leafMap[sIdx], sample->SCount(sIdx));
      bagLeaf.push_back(lb);
    }
  }
}


/**
   @brief Computes the count and rank of every bagged sample in the forest.
 */
RankCount *LeafReg::RankCounts(const std::vector<unsigned int> &row2Rank) const {
  std::vector<unsigned int> leafSeen(NodeCount());
  std::fill(leafSeen.begin(), leafSeen.end(), 0);

  RankCount *rankCount = new RankCount[BagTot()];
  BitMatrix *bag = BagRow();
  unsigned int bagIdx = 0;
  for (unsigned int tIdx = 0; tIdx < NTree(); tIdx++) {
    for (unsigned int row = 0; row < bag->NRow(); row++) {
      if (bag->TestBit(row, tIdx)) {
        unsigned int leafIdx = LeafIdx(tIdx, bagIdx);
        unsigned int bagOff = offset[leafIdx] + leafSeen[leafIdx]++;
	rankCount[bagOff].Init(row2Rank[row], SCount(bagOff));
	bagIdx++;
      }
    }
  }

  return rankCount;
}


/**
   @brief Computes bag index bounds in forest setting.
 */
void Leaf::BagBounds(unsigned int tIdx, unsigned int leafIdx, unsigned int &start, unsigned int &end) const {
  unsigned int forestIdx = NodeIdx(tIdx, leafIdx);
  start = offset[forestIdx];
  end = start + Extent(forestIdx);
}


