// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.cc

   @brief Methods for building and walking the decision forest.

   @author Mark Seligman
 */


#include "dectree.h"
#include "bv.h"
#include "forest.h"
#include "sampler.h"
#include "rleframe.h"
#include "ompthread.h"
#include "quant.h"

const size_t Forest::scoreChunk = 0x2000;
const unsigned int Forest::seqChunk = 0x20;


Forest::Forest(vector<DecTree>&& decTree_,
	       const tuple<double, double, string>& scoreDesc_,
	       Leaf&& leaf_) :
  decTree(decTree_),
  scoreDesc(ScoreDesc(scoreDesc_)),
  leaf(leaf_),
  noNode(maxHeight(decTree)),
  nTree(decTree.size()),
  idxFinal(vector<IndexT>(scoreChunk * nTree)) {
}


void Forest::initPrediction(const RLEFrame* rleFrame) {
  this->nObs = rleFrame->getNRow();
  nPredNum = rleFrame->getNPredNum();
  nPredFac = rleFrame->getNPredFac();
  
  // Ultimately handled internally.
  for (DecTree& tree : decTree)
    tree.setObsWalker(nPredNum);
}


size_t Forest::maxHeight(const vector<DecTree>& decTree) {
  size_t height = 0;
  for (const DecTree& tree : decTree) {
    height = max(height, tree.nodeCount());
  }
  return height;
}


unique_ptr<ForestPredictionReg> Forest::predictReg(const Sampler* sampler,
						   const RLEFrame* rleFrame) {
  initPrediction(rleFrame);
  unique_ptr<ForestPredictionReg> prediction = scoreDesc.makePredictionReg(this, sampler, nObs);
  predict(sampler, rleFrame, prediction.get());
  
  return prediction;
}
						   

unique_ptr<ForestPredictionCtg> Forest::predictCtg(const Sampler* sampler,
						   const RLEFrame* rleFrame) {
  initPrediction(rleFrame);
  unique_ptr<ForestPredictionCtg> prediction = scoreDesc.makePredictionCtg(this, sampler, nObs);
  predict(sampler, rleFrame, prediction.get());
  
  return prediction;
}
						   

void Forest::predict(const Sampler* sampler,
		     const RLEFrame* rleFrame,
		     ForestPrediction* prediction) {
  vector<size_t> trIdx(nPredNum + nPredFac);
  size_t row = predictBlock(sampler, rleFrame, prediction, 0, nObs, trIdx);
  // Remainder rows handled in custom-fitted block.
  if (nObs > row) {
    (void) predictBlock(sampler, rleFrame, prediction, row, nObs, trIdx);
  }
}


size_t Forest::predictBlock(const Sampler* sampler,
			    const RLEFrame* rleFrame,
			    ForestPrediction* prediction,
			    size_t rowStart,
			    size_t rowEnd,
			    vector<size_t>& trIdx) {
  size_t blockRows = min(scoreChunk, rowEnd - rowStart);
  size_t row = rowStart;
  for (; row + blockRows <= rowEnd; row += blockRows) {
    transpose(rleFrame, trIdx, row, scoreChunk);
    blockStart = row;
    predictObs(sampler, prediction, blockStart, blockRows);
  }
  
  return row;
}


void Forest::predictObs(const Sampler* sampler,
			ForestPrediction* prediction,
			size_t obsStart,
			size_t span) {
  OMPBound rowEnd = static_cast<OMPBound>(obsStart + span);
  OMPBound rowStart = static_cast<OMPBound>(obsStart);
  setBlockStart(obsStart);

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound row = rowStart; row < rowEnd; row += seqChunk) {
    walkTree(sampler, row, min(rowEnd, row + seqChunk));
    prediction->callScorer(this, row, min(rowEnd, row + seqChunk));
  }
  }
  prediction->cacheIndices(idxFinal, span * nTree, blockStart * nTree);
}


void Forest::transpose(const RLEFrame* rleFrame,
		       vector<size_t>& idxTr,
		       size_t rowStart,
		       size_t rowExtent) {
  trFac = vector<CtgT>(scoreChunk * nPredFac);
  trNum = vector<double>(scoreChunk * nPredNum);
  CtgT* facOut = trFac.empty() ? nullptr : &trFac[0];
  double* numOut = trNum.empty() ? nullptr : &trNum[0];
  for (size_t row = rowStart; row != min(nObs, rowStart + rowExtent); row++) {
    unsigned int numIdx = 0;
    unsigned int facIdx = 0;
    vector<szType> rankVec = rleFrame->idxRank(idxTr, row);
    for (unsigned int predIdx = 0; predIdx < rankVec.size(); predIdx++) {
      unsigned int rank = rankVec[predIdx];
      if (rleFrame->factorTop[predIdx] == 0) {
	*numOut++ = rleFrame->numRanked[numIdx++][rank];
      }
      else {// TODO:  Replace subtraction with (front end)::fac2Rank()
	*facOut++ = rleFrame->facRanked[facIdx++][rank] - 1;
      }
    }
  }
}


void Forest::setBlockStart(size_t blockStart) {
  this->blockStart = blockStart;
  fill(idxFinal.begin(), idxFinal.end(), noNode);
}


void Forest::walkTree(const Sampler* sampler,
		      size_t obsStart,
		      size_t obsEnd) {
  for (size_t obsIdx = obsStart; obsIdx != obsEnd; obsIdx++) {
    for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
      if (!sampler->isBagged(tIdx, obsIdx)) {
	setFinalIdx(obsIdx, tIdx, walkObs(obsIdx, tIdx));
      }
    }
  }
}


/**
   @param[out] nodeIdx is the final index of the tree walk.

   @return true iff final index is a valid node.
 */
bool Forest::getFinalIdx(size_t obsIdx, unsigned int tIdx, IndexT& nodeIdx) const {
  nodeIdx = idxFinal[nTree * (obsIdx - blockStart) + tIdx];
  return nodeIdx != noNode;
}


bool Forest::isLeafIdx(size_t obsIdx,
			unsigned int tIdx,
			IndexT& leafIdx) const {
  IndexT nodeIdx;
  if (getFinalIdx(obsIdx, tIdx, nodeIdx))
    return getLeafIdx(tIdx, nodeIdx, leafIdx);
  else
    return false;
}


bool Forest::isNodeIdx(size_t obsIdx,
		       unsigned int tIdx,
		       double& score) const {
  IndexT nodeIdx;
  if (getFinalIdx(obsIdx, tIdx, nodeIdx)) {
    score = getScore(tIdx, nodeIdx);
    return true;
  }
  else {
    return false;
  }
    // Non-bagging scenarios should always see a leaf.
    //    if (!bagging) assert(termIdx != noNode);
}


void Forest::dump(vector<vector<PredictorT> >& predTree,
                  vector<vector<double> >& splitTree,
                  vector<vector<size_t> >& delIdxTree,
		  vector<vector<double>>& scoreTree,
		  IndexT& dummy) const {
  dump(predTree, splitTree, delIdxTree, scoreTree);
}


void Forest::dump(vector<vector<PredictorT> >& pred,
                  vector<vector<double> >& split,
                  vector<vector<size_t> >& delIdx,
		  vector<vector<double>>& score) const {
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    const DecTree& tree = decTree[tIdx];
    for (IndexT nodeIdx = 0; nodeIdx < tree.nodeCount(); nodeIdx++) {
      pred[tIdx].push_back(tree.getPredIdx(nodeIdx));
      delIdx[tIdx].push_back(tree.getDelIdx(nodeIdx));
      score[tIdx].push_back(tree.getScore(nodeIdx));
      // N.B.:  split field must fit within a double.
      split[tIdx].push_back(tree.getSplitNum(nodeIdx));
    }
  }
}


vector<IndexT> Forest::getLeafNodes(unsigned int tIdx,
				    IndexT extent) const {
  vector<IndexT> leafIndices(extent);
  IndexT nodeIdx = 0;
  for (auto node : decTree[tIdx].getNode()) {
    IndexT leafIdx;
    if (node.getLeafIdx(leafIdx)) {
      leafIndices[leafIdx] = nodeIdx;
    }
    nodeIdx++;
  }

  return leafIndices;
}


vector<vector<IndexRange>> Forest::leafDominators() const {
  vector<vector<IndexRange>> leafDom(nTree);
  
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    leafDom[tIdx] = leafDominators(decTree[tIdx].getNode());
  }
  }
  return leafDom;
}
  

vector<IndexRange> Forest::leafDominators(const vector<DecNode>& tree) {
  IndexT height = tree.size();
  // Gives each node the offset of its predecessor.
  vector<IndexT> delPred(height);
  for (IndexT i = 0; i < height; i++) {
    IndexT delIdx = tree[i].getDelIdx();
    if (delIdx != 0) {
      delPred[i + delIdx] = delIdx;
      delPred[i + delIdx + 1] = delIdx + 1;
    }
  }

  // Pushes dominated leaf count up the tree.
  vector<IndexT> leavesBelow(height);
  for (IndexT i = height - 1; i > 0; i--) {
    leavesBelow[i] += (tree[i].isNonterminal() ? 0: 1);
    leavesBelow[i - delPred[i]] += leavesBelow[i];
  }

  // Pushes index ranges down the tree.
  vector<IndexRange> leafDom(height);
  leafDom[0] = IndexRange(0, leavesBelow[0]); // Root dominates all leaves.
  for (IndexT i = 0; i < height; i++) {
    IndexT delIdx = tree[i].getDelIdx();
    if (delIdx != 0) {
      IndexRange leafRange = leafDom[i];
      IndexT idxTrue = i + delIdx;
      IndexT trueStart = leafRange.getStart();
      leafDom[idxTrue] = IndexRange(trueStart, leavesBelow[idxTrue]);
      IndexT idxFalse = idxTrue + 1;
      IndexT falseStart = leafDom[idxTrue].getEnd();
      leafDom[idxFalse] = IndexRange(falseStart, leavesBelow[idxFalse]);
    }
  }

  return leafDom;
}
