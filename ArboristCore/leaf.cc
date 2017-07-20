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
   @breif Training constructor.
 */
Leaf::Leaf(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, unsigned int rowTrain) : origin(_origin), nTree(origin.size()), leafNode(_leafNode), bagLeaf(_bagLeaf), bagRow(new BitMatrix(_bagBits, rowTrain, nTree)) {
}


Leaf::~Leaf() {
  delete bagRow;
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
LeafCtg::LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagLeaf> &_bagLeaf, std::vector<unsigned int>  &_bagBits, unsigned int rowTrain, std::vector<double> &_weight, unsigned int _ctgWidth) : Leaf(_origin, _leafNode, _bagLeaf, _bagBits, rowTrain), weight(_weight), ctgWidth(_ctgWidth) {
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
  unsigned int sIdx = 0;
  for (auto row : sample2Row) {
    bagRow->SetBit(row, tIdx);
    if (!thinLeaves) {
      BagLeaf lb;
      lb.Init(leafMap[sIdx], sample->SCount(sIdx));
      sIdx++;
      bagLeaf.push_back(lb);
    }
  }
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
   @brief Static entry for recomputing tree bag count.

   @param _origin is the index of offsets into the node vector.

   @param _leafNode is the forest-wide leaf set.

   @param tIdx is the index of a tree.

   @return bag count of tree indexed by 'tIdx'.
 */
unsigned int Leaf::BagCount(const std::vector<unsigned int> &_origin, const LeafNode _leafNode[], unsigned int tIdx, unsigned int _leafCount) {
  unsigned int leafFirst = _origin[tIdx];
  unsigned int leafSup = tIdx < _origin.size() - 1 ? _origin[tIdx + 1] : _leafCount;
  unsigned int bagCount = 0;
  for (unsigned int leafIdx = leafFirst; leafIdx < leafSup; leafIdx++) {
    bagCount += _leafNode[leafIdx].Extent();
  }
  
  return bagCount;
}


/**
 */
LeafPerfReg::LeafPerfReg(const unsigned int _origin[], unsigned int _nTree, const LeafNode _leafNode[], unsigned int _leafCount, const class BagLeaf _bagLeaf[], unsigned int _bagLeafTot, unsigned int _bagBits[], unsigned int _trainRow) : LeafPerf(_origin, _nTree, _leafNode, _leafCount, _bagLeaf, _bagLeafTot, _bagBits, _trainRow), offset(std::vector<unsigned int>(leafCount)) {
  Offsets();
}


/**
   @brief Constructor for trained forest:  vector lengths final.
 */
LeafPerfCtg::LeafPerfCtg(const unsigned int _origin[], unsigned int _nTree, const class LeafNode _leafNode[], unsigned int _leafCount, const class BagLeaf _bagLeaf[], unsigned int _bagLeafTot, unsigned int _bagBits[], unsigned int _trainRow, const double _weight[], unsigned int _ctgWidth) :  LeafPerf(_origin, _nTree, _leafNode, _leafCount, _bagLeaf, _bagLeafTot, _bagBits, _trainRow), weight(_weight), ctgWidth(_ctgWidth) {
}


/**
   @brief Prediction constructor.
 */
LeafPerf::LeafPerf(const unsigned int *_origin, unsigned int _nTree, const LeafNode *_leafNode, unsigned int _leafCount, const BagLeaf *_bagLeaf, unsigned int _bagTot, unsigned int _bagBits[], unsigned int _trainRow) : origin(_origin), leafNode(_leafNode), bagLeaf(_bagLeaf), baggedRows(_bagBits == 0 ? new BitMatrix(0, 0) : new BitMatrix(_bagBits, _trainRow, _nTree)), nTree(_nTree), leafCount(_leafCount), bagLeafTot(_bagTot) {
}


LeafPerf::~LeafPerf() {
  delete baggedRows;
}


/**
   @brief Accumulates exclusive sum of counts for offset lookup.  Only
   client is quantile regression:  exits of bagLeaf[] empty.

   @return void, with side-effected reference vector.
 */
void LeafPerfReg::Offsets() {
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
void LeafPerfCtg::DefaultWeight(std::vector<double> &defaultWeight) const {
  unsigned int idx = 0;
  for (unsigned int forestIdx = 0; forestIdx < leafCount; forestIdx++) {
    for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
      defaultWeight[ctg] += weight[idx++];
    }
  }
  for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
    defaultWeight[ctg] /= leafCount;
  }
}


/**
   @brief Computes the count and rank of every bagged sample in the forest.
   Quantile regression is the only client.

   @return void.
 */
void LeafPerfReg::RankCounts(const std::vector<unsigned int> &row2Rank, std::vector<RankCount> &rankCount) const {
  if (rankCount.size() == 0)
    return;

  std::vector<unsigned int> leafSeen(leafCount);
  std::fill(leafSeen.begin(), leafSeen.end(), 0);

  unsigned int bagIdx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int row = 0; row < baggedRows->NRow(); row++) {
      if (baggedRows->TestBit(row, tIdx)) {
        unsigned int leafIdx = LeafIdx(tIdx, bagIdx);
        unsigned int bagOff = offset[leafIdx] + leafSeen[leafIdx]++;
	rankCount[bagOff].Init(row2Rank[row], SCount(bagOff));
	bagIdx++;
      }
    }
  }
}


/**
 */
void LeafReg::Export(const std::vector<unsigned int> &_origin, const LeafNode _leafNode[], unsigned int _leafCount, const BagLeaf _bagLeaf[], unsigned int _bagBits[], unsigned int _trainRow, std::vector<std::vector<unsigned int> > &rowTree, std::vector<std::vector<unsigned int> > &sCountTree, std::vector<std::vector<double> > &scoreTree, std::vector<std::vector<unsigned int> >&extentTree) {
  Leaf::Export(_origin, _leafNode, _leafCount, _bagLeaf, _bagBits, _trainRow, rowTree, sCountTree);
  LeafNode::Export(_origin, _leafNode, _leafCount, scoreTree, extentTree);
}


/**
   @brief Static exporter of BagLeaf vector into per-tree vector of vectors.

   @return void, with output reference parameters.
 */
void Leaf::Export(const std::vector<unsigned int> &_origin, const LeafNode _leafNode[], unsigned int _leafCount, const BagLeaf _bagLeaf[], unsigned int _bagBits[], unsigned int _trainRow, std::vector< std::vector<unsigned int> > &rowTree, std::vector< std::vector<unsigned int> >&sCountTree) {
  unsigned int _nTree = _origin.size();
  unsigned int bagOrig = 0;
  BitMatrix *bag = new BitMatrix(_bagBits, _trainRow, _nTree);
  for (unsigned int tIdx = 0; tIdx < _nTree; tIdx++) {
    unsigned int bagCount = BagCount(_origin, _leafNode, tIdx, _leafCount);
    rowTree[tIdx] = std::vector<unsigned int>(bagCount);
    sCountTree[tIdx] = std::vector<unsigned int>(bagCount);
    TreeExport(bag, _bagLeaf, bagOrig, tIdx, rowTree[tIdx], sCountTree[tIdx]);
    bagOrig += bagCount;
  }
  delete bag;
}


void Leaf::TreeExport(const BitMatrix *bag, const BagLeaf _bagLeaf[], unsigned int bagOrig, unsigned int tIdx, std::vector<unsigned int> &rowTree, std::vector<unsigned int> &sCountTree) {
  unsigned int bagIdx = 0;
  for (unsigned int row = 0; row < bag->NRow(); row++) {
    if (bag->TestBit(row, tIdx)) {
      rowTree[bagIdx] = row;
      sCountTree[bagIdx] = _bagLeaf[bagOrig + bagIdx].SCount();
      bagIdx++;
    }
  }
}


/**
   @brief Static exporter of LeafNode into per-tree vector of vectors.

   @param _leafCount is the count of leaves in the forest.

   @return void, with output reference parameters.
 */
void LeafNode::Export(const std::vector<unsigned int> &_origin, const LeafNode _leafNode[], unsigned int _leafCount, std::vector< std::vector<double> > &_score, std::vector< std::vector<unsigned int> > &_extent) {
  for (unsigned int tIdx = 0; tIdx < _origin.size(); tIdx++) {
    unsigned int leafCount = LeafCount(_origin, _leafCount, tIdx);
    _score[tIdx] = std::vector<double>(leafCount);
    _extent[tIdx] = std::vector<unsigned int>(leafCount);
    TreeExport(_leafNode, leafCount, _origin[tIdx], _score[tIdx], _extent[tIdx]); 
  }
}


/**
   @brief Per-tree exporter into separate vectors.
 */
void LeafNode::TreeExport(const LeafNode _leafNode[], unsigned int _leafCount, unsigned int treeOrig, std::vector<double> &scoreTree, std::vector<unsigned int> &extentTree) {
  for (unsigned int leafIdx = 0; leafIdx < _leafCount; leafIdx++) {
    _leafNode[treeOrig + leafIdx].Ref(scoreTree[leafIdx], extentTree[leafIdx]);
  }
}


/**
 */
void LeafCtg::Export(const std::vector<unsigned int> &_origin, const LeafNode _leafNode[], unsigned int _leafCount, const BagLeaf _bagLeaf[], unsigned int _bagBits[], unsigned int _trainRow, const double _weight[], unsigned int _ctgWidth, std::vector<std::vector<unsigned int> > &rowTree, std::vector<std::vector<unsigned int> > &sCountTree, std::vector<std::vector<double> > &scoreTree, std::vector<std::vector<unsigned int> > &extentTree, std::vector<std::vector<double> > &weightTree) {
  Leaf::Export(_origin, _leafNode, _leafCount, _bagLeaf, _bagBits, _trainRow, rowTree, sCountTree);
  LeafNode::Export(_origin, _leafNode, _leafCount, scoreTree, extentTree);
  for (unsigned int tIdx = 0; tIdx < _origin.size(); tIdx++) {
    unsigned int leafCount =   LeafNode::LeafCount(_origin, _leafCount, tIdx);
    weightTree[tIdx] = std::vector<double>(leafCount * _ctgWidth);
    TreeExport(_weight, _ctgWidth, _origin[tIdx] * _ctgWidth, leafCount, weightTree[tIdx]);
  }
}


void LeafCtg::TreeExport(const double leafWeight[], unsigned int _ctgWidth, unsigned int treeOrig, unsigned int leafCount, std::vector<double> &weightTree) {
  unsigned int off = 0;
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    for (unsigned int ctg = 0; ctg < _ctgWidth; ctg++) {
      weightTree[off] = leafWeight[treeOrig + off];
      off++;
    }
  }
}
