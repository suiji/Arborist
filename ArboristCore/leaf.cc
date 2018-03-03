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
#include "framemap.h"
#include "sample.h"
#include "bv.h"

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
LeafTrain::LeafTrain(unsigned int _nTree, unsigned int _rowTrain)
  :  nTree(_nTree),
    origin(vector<unsigned int>(nTree)),
    leafNode(vector<LeafNode>(0)),
    bagLeaf(vector<BagLeaf>(0)),
    bagRow(new BitMatrix(_rowTrain, nTree)) {
}


LeafTrain::~LeafTrain() {
  delete bagRow;
}


/**
 */
LeafTrainReg::LeafTrainReg(unsigned int _nTree,
		 unsigned int _rowTrain)
  : LeafTrain(_nTree, _rowTrain) {
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
LeafTrainCtg::LeafTrainCtg(unsigned int _nTree,
		 unsigned int _rowTrain,
		 unsigned int _nCtg)
  : LeafTrain(_nTree, _rowTrain),
    weight(vector<double>(0)),
    nCtg(_nCtg) {
}


LeafTrainCtg::~LeafTrainCtg() {
}


/**
   @brief Fills in leaves for a tree using current Sample.

   @param leafMap maps sampled indices to leaf indices.

   @param tIdx is the absolute tree index.

   @return void, with side-effected Leaf object.
 */
void LeafTrainReg::Leaves(const FrameTrain *frameTrain, const Sample *sample, const vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *max_element(leafMap.begin(), leafMap.end());
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
void LeafTrain::BagTree(const Sample *sample, const vector<unsigned int> &leafMap, unsigned int tIdx) {
  for (unsigned int sIdx = 0; sIdx < sample->BagCount(); sIdx++) {
    unsigned int row = sample->Sample2Row(sIdx);
    bagRow->SetBit(row, tIdx);
    if (!thinLeaves) {
      BagLeaf lb;
      lb.Init(leafMap[sIdx], sample->SCount(sIdx));
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
void LeafTrainReg::Scores(const Sample *sample, const vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx) {
  vector<unsigned int> sCount(leafCount); // Per-leaf sample counts.
  fill(sCount.begin(), sCount.end(), 0);
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
void LeafTrain::NodeExtent(const Sample *sample, vector<unsigned int> leafMap, unsigned int leafCount, unsigned int tIdx) {
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
void LeafTrainCtg::Leaves(const FrameTrain *frameTrain, const Sample *sample, const vector<unsigned int> &leafMap, unsigned int tIdx) {
  unsigned int leafCount = 1 + *max_element(leafMap.begin(), leafMap.end());
  NodeExtent(sample, leafMap, leafCount, tIdx);
  BagTree(sample, leafMap, tIdx);
  Scores(frameTrain, (SampleCtg*) sample, leafMap, leafCount, tIdx);
}


/**
   @brief Weights and scores the leaves for a classification tree.

   @param sampleCtg is the sampling vector for the current tree.
   
   @param leafMap maps sample indices to leaf indices.

   @param treeOrigin is the base leaf index of the current tree.

   @return void, with side-effected weight vector.
 */
void LeafTrainCtg::Scores(const FrameTrain *frameTrain, const SampleCtg *sample, const vector<unsigned int> &leafMap, unsigned int leafCount, unsigned int tIdx) {
  WeightInit(leafCount);

  vector<double> leafSum(leafCount);
  fill(leafSum.begin(), leafSum.end(), 0.0);
  for (unsigned int sIdx = 0; sIdx < sample->BagCount(); sIdx++) {
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
    ScoreSet(tIdx, leafIdx, argMax + maxWeight / (frameTrain->NRow() * NTree()));
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
			 unsigned int _bagBits[],
			 const double *_yTrain,
			 unsigned int _rowTrain,
			 double _meanTrain) :
  Leaf(_origin, _nTree, _leafNode, _leafCount, _bagLeaf, _bagLeafTot, _bagBits, _rowTrain),
  yTrain(_yTrain),
  meanTrain(_meanTrain),
  offset(vector<unsigned int>(leafCount)) {
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
			 unsigned int _bagBits[],
			 unsigned int _rowTrain,
			 const double _weight[],
			 unsigned int _nCtg) :
  Leaf(_origin, _nTree, _leafNode, _leafCount, _bagLeaf, _bagLeafTot, _bagBits, _rowTrain), weight(_weight), nCtg(_nCtg) {
}


/**
   @brief Prediction constructor.
 */
Leaf::Leaf(const unsigned int *_origin,
		   unsigned int _nTree,
		   const LeafNode *_leafNode,
		   unsigned int _leafCount,
		   const BagLeaf *_bagLeaf,
		   unsigned int _bagTot,
		   unsigned int _bagBits[],
		   unsigned int _rowTrain) :
  origin(_origin),
  leafNode(_leafNode),
  bagLeaf(_bagLeaf),
  rowTrain(_rowTrain),
  baggedRows(_bagBits == nullptr ? new BitMatrix(0, 0) : new BitMatrix(_bagBits, rowTrain, _nTree)),
  nTree(_nTree),
  leafCount(_leafCount),
  bagLeafTot(_bagTot) {
}


Leaf::~Leaf() {
  delete baggedRows;
}


/**
   @brief Accumulates exclusive sum of counts for offset lookup.  Only
   client is quantile regression:  exits of bagLeaf[] empty.

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
    for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
      defaultWeight[ctg] += weight[idx++];
    }
  }
  for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
    defaultWeight[ctg] /= leafCount;
  }
}


/**
   @brief Computes the count and rank of every bagged sample in the forest.
   Quantile regression is the only client.

   @return void.
 */
void LeafReg::RankCounts(const vector<unsigned int> &row2Rank, vector<RankCount> &rankCount) const {
  if (rankCount.size() == 0)
    return;

  vector<unsigned int> leafSeen(leafCount);
  fill(leafSeen.begin(), leafSeen.end(), 0);

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
void LeafReg::Export(unsigned int &_rowTrain,
			 vector<vector<unsigned int> > &rowTree,
			 vector<vector<unsigned int> > &sCountTree,
			 vector<vector<double> > &scoreTree,
			 vector<vector<unsigned int> >&extentTree) const {
  _rowTrain = rowTrain;
  Leaf::Export(rowTree, sCountTree);
  NodeExport(scoreTree, extentTree);
}


size_t LeafTrain::NodeBytes() const {
  return leafNode.size() * sizeof(LeafNode);
}


size_t LeafTrain::BLBytes() const {
  return bagLeaf.size() * sizeof(BagLeaf);
}


size_t LeafTrain::BagBytes() const {
  return bagRow->Bytes();
}


/**
   @brief Derives number of training rows from associated bag.

   @return Count of rows uniquely sampled.
 */
unsigned int LeafTrain::RowTrain() const {
  return bagRow->NRow();
}
  


/** 
    @brief Serializes the internally-typed objects, 'LeafNode', as well
    as the unsigned integer (packed bit) vector, "bagBits".
*/
void LeafTrain::Serialize(unsigned char *leafRaw, unsigned char *blRaw, unsigned char*bbRaw) const {
  for (size_t i = 0; i < NodeBytes(); i++) {
    leafRaw[i] = ((unsigned char*) &leafNode[0])[i];
  }

  for (size_t i = 0; i < BLBytes(); i++) {
    blRaw[i] = ((unsigned char*) &bagLeaf[0])[i];
  }

  bagRow->Serialize(bbRaw);
}


/**
   @brief Static exporter of BagLeaf vector into per-tree vector of vectors.

   @return void, with output reference parameters.
 */
void Leaf::Export(vector< vector<unsigned int> > &rowTree,
		      vector< vector<unsigned int> > &sCountTree) const {
  unsigned int leafOff = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int row = 0; row < rowTrain; row++) {
      if (baggedRows->TestBit(row, tIdx)) {
        rowTree[tIdx].push_back(row);
        sCountTree[tIdx].push_back(bagLeaf[leafOff++].SCount());
      }
    }
  }
}


/**
   @brief Exports LeafNode into vectors of per-tree vectors.



   @return void, with output reference parameters.
 */
void Leaf::NodeExport(vector<vector<double> > &score,
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
 */
void LeafCtg::Export(unsigned int &_rowTrain,
		     vector<vector<unsigned int> > &rowTree,
		     vector<vector<unsigned int> > &sCountTree,
		     vector<vector<double> > &scoreTree,
		     vector<vector<unsigned int> > &extentTree,
		     vector<vector<double> > &weightTree) const {
  _rowTrain = rowTrain;
  Leaf::Export(rowTree, sCountTree);
  NodeExport(scoreTree, extentTree);

  unsigned int off = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int leafIdx = 0; leafIdx < TreeHeight(tIdx); leafIdx++) {
      for (unsigned int ctg = 0; ctg < nCtg; ctg++) {
	weightTree[tIdx].push_back(weight[off++]);
      }
    }
  }
}
