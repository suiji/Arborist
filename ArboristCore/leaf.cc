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
#include "forest.h"

Leaf::Leaf(const Forest *_forest) : forest(_forest) {
}


unsigned int Leaf::TreeHeight(int tIdx) {
  return forest->TreeHeight(tIdx);
}


void Leaf::ScoreAccum(int tIdx, int leafIdx, double incr) {
  forest->ScoreAccum(tIdx, leafIdx, incr);
}


void Leaf::ScoreReg(int tIdx, int leafIdx, unsigned int _sCount) {
  forest->ScoreReg(tIdx, leafIdx, _sCount);
}


void Leaf::LeafAccum(int tIdx, unsigned int off) {
  forest->LeafAccum(tIdx, off);
}


bool Leaf::Nonterminal(int tIdx, unsigned int off) {
  return forest->Nonterminal(tIdx, off);
}


void Leaf::ScoreCtg(int tIdx, unsigned int off, unsigned int ctg, double wt) {
  forest->ScoreCtg(tIdx, off, ctg, wt);
}


int *Leaf::ExtentPosition(int tIdx) {
  return forest->ExtentPosition(tIdx);
}


/**
 */
LeafReg::LeafReg(const Forest *_forest, std::vector<unsigned int> &_sCount, std::vector<unsigned int> &_rank, int bagEst) : Leaf(_forest), sCount(_sCount), rank(_rank) {
  sCount.reserve(bagEst);
  rank.reserve(bagEst);
}


/**
 */
LeafCtg::LeafCtg(const Forest *_forest, std::vector<double> &_weight, unsigned int _ctgWidth, int heightEst) : Leaf(_forest), weight(_weight), ctgWidth(_ctgWidth) {
  weight.reserve(heightEst * ctgWidth);
}


/**
   @brief Derives and copies regression leaf information.

   @param nonTerm is zero iff forest index is at leaf.

   @param leafExtent gives leaf width at forest index.

   @param rank outputs leaf ranks; vector length bagCount.

   @param sCount outputs sample counts; vector length bagCount.

   @return bag count, with output parameter vectors.
 */
void LeafReg::Leaves(const SampleReg *sampleReg, const unsigned int frontierMap[], int tIdx) {
  Scores(sampleReg, frontierMap, tIdx);

  unsigned int bagCount = sampleReg->BagCount();
  Extent(frontierMap, bagCount, tIdx);
  int *extentPos = ExtentPosition(tIdx);

  std::vector<unsigned int> seen(TreeHeight(tIdx));
  std::fill(seen.begin(), seen.end(), 0);
  int bagOrig = rank.size();
  rank.insert(rank.end(), bagCount, 0);
  sCount.insert(sCount.end(), bagCount, 0);
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    int leafIdx = frontierMap[sIdx];
    int rkOff = extentPos[leafIdx] + seen[leafIdx]++;
    sCount[bagOrig + rkOff] = sampleReg->SCount(sIdx);
    rank[bagOrig + rkOff] = sampleReg->Rank(sIdx);
  }

  delete [] extentPos;
}


/**
   @brief Derives scores for regression tree:  intialize, accumulate, divide.

   @param frontierMap maps sample id to pretree terminal id.

   @param treeHeight is the number of nodes in the pretree.

   @return void, with output parameter vector.
*/
void LeafReg::Scores(const SampleReg *sampleReg, const unsigned int frontierMap[], int tIdx) {
  unsigned int treeHeight = TreeHeight(tIdx);
  std::vector<unsigned int> sCTree(treeHeight);
  std::fill(sCTree.begin(), sCTree.end(), 0);

  // Terminals retain initial 0.0 value; only nonterminals assigned so far.
  //
  for (unsigned int sIdx = 0; sIdx < sampleReg->BagCount(); sIdx++) {
    int leafIdx = frontierMap[sIdx];
    ScoreAccum(tIdx, leafIdx, sampleReg->Sum(sIdx));
    sCTree[leafIdx] += sampleReg->SCount(sIdx);
  }

  for (unsigned int ptIdx = 0; ptIdx < treeHeight; ptIdx++) {
    if (sCTree[ptIdx] > 0) {
      ScoreReg(tIdx, ptIdx, sCTree[ptIdx]);
    }
  }
}


/**
   @brief Sets sample counts on each leaf of a single tree.

   @param frontierMap maps sample indices to their frontier tree positions.

   @param bagCount is the number of sampled row indices.

   @param tIdx is the tree index.

   @return void with side-effected forest terminals.
 */
void Leaf::Extent(const unsigned int frontierMap[], unsigned int bagCount, int tIdx) {
  for (unsigned int i = 0; i < bagCount; i++) {
    LeafAccum(tIdx, frontierMap[i]);
  }
}


/**
   @brief Computes leaf weights and scores for a classification tree.

   @return void, with side-effected weights and forest terminals.
 */
void LeafCtg::Leaves(const SampleCtg *sampleCtg, const unsigned int frontierMap[], int tIdx) {
  Extent(frontierMap, sampleCtg->BagCount(), tIdx);
  Scores(sampleCtg, frontierMap, tIdx);
}


/**
   @brief Weights and scores the leaves for a classification tree.

   @param sampleCtg is the sampling vector for the current tree.
   
   @param frontierMap maps sample indices to terminal tree nodes.

   @param tIdx is the index of the current tree.

   @return void, with side-effected weight vector.
 */
void LeafCtg::Scores(const SampleCtg *sampleCtg, const unsigned int frontierMap[], int tIdx) {
  unsigned int treeHeight = TreeHeight(tIdx);
  int leafOff = weight.size();  // Forest offset for leaf.
  weight.insert(weight.end(), ctgWidth * treeHeight, 0.0);
  double *leafBase = &weight[leafOff];

  std::vector<double> leafSum(treeHeight);
  std::fill(leafSum.begin(), leafSum.end(), 0.0);
  for (unsigned int i = 0; i < sampleCtg->BagCount(); i++) {
    unsigned int leafIdx = frontierMap[i];
    FltVal sum;
    unsigned int dummy;
    unsigned int ctg = sampleCtg->Ref(i, sum, dummy);
    leafSum[leafIdx] += sum;
    leafBase[leafIdx * ctgWidth + ctg] += sum;
  }

  // Normalizes weights for probabilities.
  for (unsigned int leafIdx = 0; leafIdx < treeHeight; leafIdx++) {
    if (!Nonterminal(tIdx, leafIdx)) {
      double maxWeight = 0.0;
      unsigned int argMax = 0;
      double recipSum = 1.0 / leafSum[leafIdx];
      for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
	double thisWeight = leafBase[leafIdx * ctgWidth + ctg] * recipSum;
	leafBase[leafIdx * ctgWidth + ctg] = thisWeight;
	if (thisWeight > maxWeight) {
	  maxWeight = thisWeight;
	  argMax = ctg;
	}
      }
      ScoreCtg(tIdx, leafIdx, argMax, maxWeight);
    }
  }
}
