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


/**
   @brief Reserves leafNode space based on estimate.

   @return void.
 */
void Leaf::Reserve(unsigned int leafEst) {
  leafNode.reserve(leafEst * nTree);
}


/**
   @brief Reserves space based on leaf- and bag-count estimates.

   @return void.
 */
void LeafReg::Reserve(unsigned int leafEst, unsigned int bagEst) {
  Leaf::Reserve(leafEst);
  info.reserve(bagEst * NTree());
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


/**
 */
void LeafCtg::Reserve(unsigned int leafEst) {
  Leaf::Reserve(leafEst);
  info.reserve(leafEst * ctgWidth * NTree());
}


/**
   @brief Derives and copies regression leaf information.

   @param leafExtent gives leaf width at forest index.

   @param rank outputs leaf ranks; vector length bagCount.

   @return bag count, with output parameter vectors.
 */
void LeafReg::Leaves(const SampleReg *sampleReg, const std::vector<unsigned int> &frontierMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *std::max_element(frontierMap.begin(), frontierMap.end());
  NodeExtent(frontierMap, sampleReg->BagCount(), leafCount, tIdx);
  Scores(sampleReg, frontierMap, leafCount, tIdx);
  SampleInfo(sampleReg, frontierMap, leafCount, tIdx);
}


/**
   @brief Derives scores for regression tree:  intialize, accumulate, divide.

   @param frontierMap maps sample id to leaf index.

   @param leafCount is the number of leaves in the tree.

   @return void, with output parameter vector.
*/
void LeafReg::Scores(const SampleReg *sampleReg, const std::vector<unsigned int> &frontierMap, unsigned int leafCount, unsigned int tIdx) {
  std::vector<unsigned int> sCount(leafCount); // Leaf sample counts.
  std::fill(sCount.begin(), sCount.end(), 0);
  for (unsigned int sIdx = 0; sIdx < sampleReg->BagCount(); sIdx++) {
    unsigned int leafIdx = frontierMap[sIdx];
    ScoreAccum(tIdx, leafIdx, sampleReg->Sum(sIdx));
    sCount[leafIdx] += sampleReg->SCount(sIdx);
  }

  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    ScoreScale(tIdx, leafIdx, sCount[leafIdx]);
  }
}


void LeafReg::SampleInfo(const SampleReg *sampleReg, const std::vector<unsigned int> &frontierMap, unsigned int leafCount, unsigned int tIdx) {  
  std::vector<unsigned int> sampleOffset(leafCount);
  SampleOffset(sampleOffset, Origin(tIdx), leafCount, info.size());

  std::vector<unsigned int> leafSeen(leafCount);
  std::fill(leafSeen.begin(), leafSeen.end(), 0);
  RankCount init;
  init.Init();
  unsigned int bagCount = sampleReg->BagCount();
  info.insert(info.end(), bagCount, init);
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    unsigned int leafIdx = frontierMap[sIdx];
    InfoSet(sampleOffset[leafIdx] + leafSeen[leafIdx]++, sampleReg->SCount(sIdx), sampleReg->Rank(sIdx));
  }
}


/**
   @brief Writes the current tree origin and computes the extent of each leaf node.

   @param leafCount is the number of leaves in the current tree.

   @void, with count-adjusted leaf nodes.
 */
void Leaf::NodeExtent(std::vector<unsigned int> frontierMap, unsigned int bagCount, unsigned int leafCount, unsigned int tIdx) {
  unsigned int leafBase = leafNode.size();
  origin[tIdx] = leafBase;

  LeafNode init;
  init.Init();
  leafNode.insert(leafNode.end(), leafCount, init);
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    unsigned int leafIdx = frontierMap[sIdx];
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
void LeafCtg::Leaves(const SampleCtg *sampleCtg, const std::vector<unsigned int> &frontierMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *std::max_element(frontierMap.begin(), frontierMap.end());
  NodeExtent(frontierMap, sampleCtg->BagCount(), leafCount, tIdx);
  Scores(sampleCtg, frontierMap, leafCount, tIdx);
}


/**
   @brief Weights and scores the leaves for a classification tree.

   @param sampleCtg is the sampling vector for the current tree.
   
   @param frontierMap maps sample indices to terminal tree nodes.

   @param treeOrigin is the base leaf index of the current tree.

   @return void, with side-effected weight vector.
 */
void LeafCtg::Scores(const SampleCtg *sampleCtg, const std::vector<unsigned int> &frontierMap, unsigned int leafCount, unsigned int tIdx) {
  WeightInit(leafCount);

  std::vector<double> leafSum(leafCount);
  std::fill(leafSum.begin(), leafSum.end(), 0.0);
  for (unsigned int sIdx = 0; sIdx < sampleCtg->BagCount(); sIdx++) {
    unsigned int leafIdx = frontierMap[sIdx];
    FltVal sum;
    unsigned int dummy;
    unsigned int ctg = sampleCtg->Ref(sIdx, sum, dummy);
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
