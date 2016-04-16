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

#include <algorithm>
using namespace std;

//#include <iostream>

Leaf::Leaf(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode) : origin(_origin), nTree(origin.size()), leafNode(_leafNode) {
}


/**
 */
LeafReg::LeafReg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<RankCount> &_info) : Leaf(_origin, _leafNode),  info(_info) {
}


LeafReg::~LeafReg() {
}


/**
   @brief Reserves leafNode space based on estimate.

   @return void.
 */
void Leaf::Reserve(unsigned int leafEst) {
  leafNode.reserve(leafEst);
}


/**
   @brief Reserves space based on leaf- and bag-count estimates.

   @return void.
 */
void LeafReg::Reserve(unsigned int leafEst, unsigned int bagEst) {
  Leaf::Reserve(leafEst);
  info.reserve(bagEst);
}


/**
 */
void LeafCtg::Reserve(unsigned int leafEst, unsigned int bagEst) {
  Leaf::Reserve(leafEst);
  info.reserve(leafEst * ctgWidth);
}


/**
   @brief Constructor for incipient forest.
 */
LeafCtg::LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<double> &_info, unsigned int _ctgWidth) : Leaf(_origin, _leafNode), info(_info), ctgWidth(_ctgWidth) {
}


/**
   @brief Constructor for trained forest:  vector lengths final.
 */
LeafCtg::LeafCtg(std::vector<unsigned int> &_origin, std::vector<LeafNode> &_leafNode, std::vector<double> &_info) : Leaf(_origin, _leafNode), info(_info), ctgWidth(info.size() / NodeCount()) {
}


LeafCtg::~LeafCtg() {
}


/**
   @brief Derives and copies regression leaf information.

   @param leafExtent gives leaf width at forest index.

   @param rank outputs leaf ranks; vector length bagCount.

   @return bag count, with output parameter vectors.
 */
void LeafReg::Leaves(const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *std::max_element(leafMap.begin(), leafMap.end());
  NodeExtent(leafMap, sample->BagCount(), leafCount, tIdx);
  Scores(sample, leafMap, leafCount, tIdx);
  SampleInfo((SampleReg *) sample, leafMap, leafCount, tIdx);
}


/**
   @brief Derives scores for regression tree:  intialize, accumulate, divide.

   @param leafMap maps sample id to leaf index.

   @param leafCount is the number of leaves in the tree.

   @return void, with output parameter vector.
*/
void LeafReg::Scores(const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx) {
  std::vector<unsigned int> sCount(leafCount); // Leaf sample counts.
  std::fill(sCount.begin(), sCount.end(), 0);
  for (unsigned int sIdx = 0; sIdx < sample->BagCount(); sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    ScoreAccum(tIdx, leafIdx, sample->Sum(sIdx));
    sCount[leafIdx] += sample->SCount(sIdx);
  }

  // TODO:
  //  i) Move sCount[] to base class
  // ii) Replace RankCount vector with vector of uint.
  //iii) Introduce uint row[] vector, parallel to sCount[], also in base.
  // iv) Invert Sample's row2Sample[] to construct (iii); size = bagCount.

  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    ScoreScale(tIdx, leafIdx, sCount[leafIdx]);
  }
}


void LeafReg::SampleInfo(const SampleReg *sample, const std::vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx) {  
  std::vector<unsigned int> sampleOffset(leafCount);
  SampleOffset(sampleOffset, Origin(tIdx), leafCount, info.size());

  std::vector<unsigned int> leafSeen(leafCount);
  std::fill(leafSeen.begin(), leafSeen.end(), 0);
  RankCount init;
  init.Init();
  unsigned int bagCount = sample->BagCount();
  info.insert(info.end(), bagCount, init);
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    InfoSet(sampleOffset[leafIdx] + leafSeen[leafIdx]++, sample->SCount(sIdx), sample->Rank(sIdx));
  }
}


/**
   @brief Writes the current tree origin and computes the extent of each leaf node.

   @param leafCount is the number of leaves in the current tree.

   @void, with count-adjusted leaf nodes.
 */
void Leaf::NodeExtent(std::vector<unsigned int> leafMap, unsigned int bagCount, unsigned int leafCount, unsigned int tIdx) {
  unsigned int leafBase = leafNode.size();
  origin[tIdx] = leafBase;

  LeafNode init;
  init.Init();
  leafNode.insert(leafNode.end(), leafCount, init);
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    unsigned int leafIdx = leafMap[sIdx];
    leafNode[leafBase + leafIdx].Count()++;
  }
}


/**
   @brief Accumulates exclusive sum of counts for offset lookup.

   @param sampleOffset outputs accumulated counts.

   @param treeOffset is the forest-based offset of the current tree, i.e,
   the accumulated bag counts of all preceding trees.  Default value of
   zero used for whole-forest access.

   @return void, with output reference vector.
 */
void LeafReg::SampleOffset(std::vector<unsigned int> &sampleOffset, unsigned int leafBase, unsigned int leafCount, unsigned int sampleBase) const {
  unsigned int countAccum = 0;
  for (unsigned int leafIdx = leafBase; leafIdx < leafBase + leafCount; leafIdx++) {
    sampleOffset[leafIdx - leafBase] = sampleBase + countAccum;
    countAccum += Extent(leafIdx);
  }
}


/**
   @brief Computes leaf weights and scores for a classification tree.

   @return void, with side-effected weights and forest terminals.
 */
void LeafCtg::Leaves(const Sample *sample, const std::vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *std::max_element(leafMap.begin(), leafMap.end());
  NodeExtent(leafMap, sample->BagCount(), leafCount, tIdx);
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
   @brief Static exporter into per-tree vector of vectors.

   @return void, with output reference parameters.
 */
void LeafNode::Export(const std::vector<unsigned int> &_leafOrigin, const std::vector<LeafNode> &_leafNode, std::vector< std::vector<double> > &_score, std::vector< std::vector<unsigned int> > &_extent) {
  for (unsigned int tIdx = 0; tIdx < _leafOrigin.size(); tIdx++) {
    unsigned int leafCount = LeafCount(_leafOrigin, _leafNode.size(), tIdx);
    _score[tIdx] = std::vector<double>(leafCount);
    _extent[tIdx] = std::vector<unsigned int>(leafCount);
    TreeExport(_leafNode, _leafOrigin[tIdx], leafCount, _score[tIdx], _extent[tIdx]); 
  }
}


/**
   @brief Per-tree exporter into separate vectors.
 */
void LeafNode::TreeExport(const std::vector<LeafNode> &_leafNode, unsigned int treeOff, unsigned int leafCount, std::vector<double> &_score, std::vector<unsigned int> &_extent) {
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    _leafNode[treeOff + leafIdx].Ref(_score[leafIdx], _extent[leafIdx]);
  }
}


/**
 */
void LeafReg::Export(const std::vector<unsigned int> &_leafOrigin, const std::vector<RankCount> &_leafInfo, std::vector< std::vector<unsigned int> > &_rank, std::vector< std::vector<unsigned int> > &_sCount) {
  for (unsigned int tIdx = 0; tIdx < _leafOrigin.size(); tIdx++) {
    unsigned int leafCount = LeafCount(_leafOrigin, _leafInfo.size(), tIdx);
    _rank[tIdx] = std::vector<unsigned int>(leafCount);
    _sCount[tIdx] = std::vector<unsigned int>(leafCount);
    TreeExport(_leafInfo, _leafOrigin[tIdx], leafCount,  _rank[tIdx], _sCount[tIdx]);
  }
}


/**
 */
unsigned int LeafReg::LeafCount(const std::vector<unsigned int> &_origin, unsigned int height, unsigned int tIdx) {
  return LeafNode::LeafCount(_origin, height, tIdx);
}


void LeafReg::TreeExport(const std::vector<RankCount> &_rankCount, unsigned int treeOff, unsigned int leafCount, std::vector<unsigned int> &_rank, std::vector<unsigned int> &_sCount) {
  _rank.reserve(leafCount);
  _sCount.reserve(leafCount);
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    _rankCount[treeOff + leafIdx].Ref(_rank[leafIdx], _sCount[leafIdx]);
  }
}


/**
 */
void LeafCtg::Export(const std::vector<unsigned int> &_leafOrigin, const std::vector<double> &_leafInfo, unsigned int _ctgWidth, std::vector< std::vector<double> > &_weight) {
  for (unsigned int tIdx = 0; tIdx < _leafOrigin.size(); tIdx++) {
    unsigned int leafCount = LeafCount(_leafOrigin, _leafInfo.size(), _ctgWidth, tIdx);
    _weight[tIdx] = std::vector<double>(leafCount * _ctgWidth);
    TreeExport(_leafInfo, _ctgWidth, _leafOrigin[tIdx] * _ctgWidth, leafCount, _weight[tIdx]);
  }
}


unsigned int LeafCtg::LeafCount(std::vector<unsigned int> _origin, unsigned int infoLen, unsigned int _ctgWidth, unsigned int tIdx) {
  return LeafNode::LeafCount(_origin, infoLen / _ctgWidth, tIdx);
}


void LeafCtg::TreeExport(const std::vector<double> &leafInfo, unsigned int _ctgWidth, unsigned int treeOff, unsigned int leafCount, std::vector<double> &_weight) {
  unsigned int off = 0;
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    for (unsigned int ctg = 0; ctg < _ctgWidth; ctg++) {
      _weight[off] = leafInfo[treeOff + off];
      off++;
    }
  }
}
