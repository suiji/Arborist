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
using namespace std;

//#include <iostream>

Leaf::Leaf(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow) : origin(_origin), nTree(origin.size()), leafNode(_leafNode), bagRow(_bagRow) {
}


/**
 */
LeafReg::LeafReg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<unsigned int> &_rank) : Leaf(_origin, _leafNode, _bagRow),  rank(_rank) {
}


LeafReg::~LeafReg() {
}


/**
   @brief Reserves leafNode space based on estimate.

   @return void.
 */
void Leaf::Reserve(unsigned int leafEst, unsigned int bagEst) {
  leafNode.reserve(leafEst);
  bagRow.reserve(bagEst);
}


/**
   @brief Reserves space based on leaf- and bag-count estimates.

   @return void.
 */
void LeafReg::Reserve(unsigned int leafEst, unsigned int bagEst) {
  Leaf::Reserve(leafEst, bagEst);
  rank.reserve(bagEst);
}


/**
 */
void LeafCtg::Reserve(unsigned int leafEst, unsigned int bagEst) {
  Leaf::Reserve(leafEst, bagEst);
  weight.reserve(leafEst * ctgWidth);
}


/**
   @brief Constructor for incipient forest.
 */
LeafCtg::LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<double> &_weight, unsigned int _ctgWidth) : Leaf(_origin, _leafNode, _bagRow), weight(_weight), ctgWidth(_ctgWidth) {
}


/**
   @brief Constructor for trained forest:  vector lengths final.
 */
LeafCtg::LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<BagRow> &_bagRow, std::vector<double> &_weight) : Leaf(_origin, _leafNode, _bagRow), weight(_weight), ctgWidth(weight.size() / NodeCount()) {
}


LeafCtg::~LeafCtg() {
}


/**
   @brief Builds a bit matrix for the forest bag set.  Each row/column pair is
   read exactly once during validation, so greatest benefits lie in iterative
   workflows, such as importance permutation.

   @param bagTrain is the number of rows used to train or zero, if not using bag.

   @return bagged bit matrix.
 */
BitMatrix *Leaf::ForestBag(unsigned int bagTrain) {
  if (bagTrain == 0) // Not using bag.
    return new BitMatrix(0, 0);
  
  unsigned int nTree = origin.size();
  BitMatrix *forestBag = new BitMatrix(bagTrain, nTree); 
  unsigned int sIdx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    unsigned int bagCount = BagCount(origin, leafNode, tIdx);
    for (unsigned int idx = 0; idx < bagCount; idx++) {
      forestBag->SetBit(bagRow[sIdx++].Row(), tIdx);
    }
  }

  return forestBag;
}


/**
   @brief Derives and copies regression leaf information.

   @param leafExtent gives leaf width at forest index.

   @param rank outputs leaf ranks; vector length bagCount.

   @return bag count, with output parameter vectors.
 */
void LeafReg::Leaves(const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *std::max_element(leafMap.begin(), leafMap.end());
  NodeExtent(sample, leafMap, leafCount, tIdx);
  RowBag(sample, leafMap, leafCount, tIdx);
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
   @brief Records bagged rows, by index, and sets any parallel auxilliary vectors.

   @param sample is the sampling record for the current tree.

   @param leafMap maps sample indices to their frontier (leaf) positions.

   @param bagCount is the

   @return void.
 */
void Leaf::RowBag(const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx) {
  unsigned int bagCount = sample->BagCount();
  std::vector<unsigned int> sample2Row(bagCount);
  sample->RowInvert(sample2Row);
  
  std::vector<unsigned int> sampleOffset(leafCount);
  SampleOffset(sampleOffset, Origin(tIdx), leafCount, bagRow.size());

  BagRow brInit;
  brInit.Init();
  bagRow.insert(bagRow.end(), bagCount, brInit);
  RankInit(bagCount, 0);

  std::vector<unsigned int> leafSeen(leafCount);
  std::fill(leafSeen.begin(), leafSeen.end(), 0);
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    unsigned int sOff = sampleOffset[leafIdx] + leafSeen[leafIdx]++;
    bagRow[sOff].Set(sample2Row[sIdx], sample->SCount(sIdx));
    RankSet(sOff, sample, sIdx);
  }
  // post-condition:  sum(leafSeen) == bagCount
}


void LeafReg::RankInit(unsigned int bagCount, unsigned int init) {
  rank.insert(rank.end(), bagCount, 0);
}


void LeafReg::RankSet(unsigned int sOff, const class Sample *sample, unsigned int sIdx) {
  rank[sOff] = ((SampleReg *) sample)->Rank(sIdx);
}




/**
   @brief Accumulates exclusive sum of counts for offset lookup.

   @param sampleOffset outputs accumulated counts into tree-based indices.

   @param bagEnd is the next available index in the accumulated bag.  Default value
   of zero implies whole-forest access.

   @return void, with output reference vector.
 */
void Leaf::SampleOffset(std::vector<unsigned int> &sampleOffset, unsigned int leafBase, unsigned int leafCount, unsigned int bagEnd) const {
  unsigned int countAccum = 0;
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    sampleOffset[leafIdx] = bagEnd + countAccum;
    countAccum += Extent(leafBase + leafIdx);
  }
  // Post-condition:  countAccum == bagCount
}


/**
   @brief Computes leaf weights and scores for a classification tree.

   @return void, with side-effected weights and forest terminals.
 */
void LeafCtg::Leaves(const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *std::max_element(leafMap.begin(), leafMap.end());
  NodeExtent(sample, leafMap, leafCount, tIdx);
  RowBag(sample, leafMap, leafCount, tIdx);
  Scores((SampleCtg*) sample, leafMap, leafCount, tIdx);
}


/**
   @brief Weights and scores the leaves for a classification tree.

   @param sampleCtg is the sampling vector for the current tree.
   
   @param leafMap maps sample indices to leaf indices.

   @param treeOrigin is the base leaf index of the current tree.

   @return void, with side-effected weight vector.
 */
void LeafCtg::Scores(const SampleCtg *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx) {
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
    ScoreSet(tIdx, leafIdx, argMax + maxWeight / (PredBlock::NRow() * NTree()));
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
void LeafReg::Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, const vector<BagRow> &_bagRow, const std::vector<unsigned int> &_rank, std::vector<std::vector<unsigned int> > &rowTree, std::vector<std::vector<unsigned int> > &sCountTree, std::vector<std::vector<double> > &scoreTree, std::vector<std::vector<unsigned int> >&extentTree, std::vector< std::vector<unsigned int> > &rankTree) {
  Leaf::Export(_origin, _leafNode, _bagRow, rowTree, sCountTree);
  LeafNode::Export(_origin, _leafNode, scoreTree, extentTree);
  unsigned int bagOrig = 0;
  for (unsigned int tIdx = 0; tIdx < _origin.size(); tIdx++) {
    unsigned int bagCount = BagCount(_origin, _leafNode, tIdx);
    rankTree[tIdx] = std::vector<unsigned int>(bagCount);
    TreeExport(_rank, bagOrig, bagCount, rankTree[tIdx]);
    bagOrig += bagCount;
  }
}


/**
   @brief Static exporter of BagRow vector into per-tree vector of vectors.

   @return void, with output reference parameters.
 */
void Leaf::Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, const std::vector<BagRow> &_bagRow, std::vector< std::vector<unsigned int> > &rowTree, std::vector< std::vector<unsigned int> >&sCountTree) {
  unsigned int nTree = _origin.size();
  unsigned int bagOrig = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    unsigned int bagCount = BagCount(_origin, _leafNode, tIdx);
    rowTree[tIdx] = std::vector<unsigned int>(bagCount);
    sCountTree[tIdx] = std::vector<unsigned int>(bagCount);
    TreeExport(_bagRow, bagOrig, bagCount, rowTree[tIdx], sCountTree[tIdx]);
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


void Leaf::TreeExport(const std::vector<BagRow> &_bagRow, unsigned int bagOrig, unsigned int bagCount, std::vector<unsigned int> &rowTree, std::vector<unsigned int> &sCountTree) {
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    _bagRow[bagOrig + sIdx].Ref(rowTree[sIdx], sCountTree[sIdx]);
  }
}


/**
   @brief Copies sections of forest-wide fields to per-tree vectors.

   @return void.
 */
void LeafReg::TreeExport(const std::vector<unsigned int> &_rank, unsigned int bagOrig, unsigned int bagCount, std::vector<unsigned int> &rankTree) {
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    rankTree[sIdx] = _rank[bagOrig + sIdx];
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
void LeafCtg::Export(const std::vector<unsigned int> &_origin, const std::vector<LeafNode> &_leafNode, const std::vector<BagRow> &_bagRow, const std::vector<double> &_weight, unsigned int _ctgWidth, std::vector<std::vector<unsigned int> > &rowTree, std::vector<std::vector<unsigned int> > &sCountTree, std::vector<std::vector<double> > &scoreTree, std::vector<std::vector<unsigned int> > &extentTree, std::vector<std::vector<double> > &weightTree) {
  Leaf::Export(_origin, _leafNode, _bagRow, rowTree, sCountTree);
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
