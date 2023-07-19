// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file pretree.cc

   @brief Class defintions for the pre-tree, a serial and minimal representation from which the decision tree is built.

   @author Mark Seligman
 */
#include "train.h"
#include "predictorframe.h"
#include "indexset.h"
#include "leaf.h"
#include "samplemap.h"
#include "bv.h"
#include "splitfrontier.h"
#include "pretree.h"

#include <queue>
#include <vector>

IndexT PreTree::leafMax = 0;


PreTree::PreTree(const PredictorFrame* frame,
		 IndexT bagCount) :
  leafCount(0),
  infoLocal(vector<double>(frame->getNPred())),
  splitBits(BV(bagCount * frame->getFactorExtent())), // Vague estimate.
  observedBits(BV(bagCount * frame->getFactorExtent())),
  bitEnd(0) {
}


void PreTree::init(IndexT leafMax_) {
  leafMax = leafMax_;
}



void PreTree::deInit() {
  leafMax = 0;
}


void PreTree::consumeCompound(const SplitFrontier* sf,
			      const vector<vector<SplitNux>>& nuxMax) {
  // True branches target box exterior.
  // False branches target next criterion or box terminal.
  for (auto & nuxCrit : nuxMax) {
    consumeCriteria(sf, nuxCrit);
  }
}


void PreTree::consumeCriteria(const SplitFrontier* sf,
			      const vector<SplitNux>& nuxCrit) {
  offspring(nuxCrit.size()); // Preallocates terminals and compound nonterminals.
  for (auto nux : nuxCrit) {
    addCriterion(sf, nux, true);
  }
}


void PreTree::addCriterion(const SplitFrontier* sf,
			   const SplitNux& nux,
			   bool preallocated) {
  if (nux.noNux())
    return;

  if (sf->isFactor(nux)) {
    critBits(sf, nux);
  }
  else {
    critCut(sf, nux);
  }

  offspring(preallocated ? 0 : 1);
  DecNode& node = getNode(nux.getPTId());
  node.setInvert(nux.invertTest());
  node.setDelIdx(getHeight() - 2 - nux.getPTId());
  infoNode[nux.getPTId()] = nux.getInfo();
  infoLocal[node.getPredIdx()] += nux.getInfo();
}


void PreTree::critBits(const SplitFrontier* sf,
		       const SplitNux& nux) {
  auto bitPos = bitEnd;
  splitBits.resize(bitEnd);
  observedBits.resize(bitEnd);
  bitEnd += sf->critBitCount(nux);
  sf->setTrueBits(nux, &splitBits, bitPos);
  sf->setObservedBits(nux, &observedBits, bitPos);
  getNode(nux.getPTId()).critBits(nux, bitPos);
}


void PreTree::critCut(const SplitFrontier* sf,
		      const SplitNux& nux) {
  getNode(nux.getPTId()).critCut(nux, sf);
}


void PreTree::setScore(const IndexSet& iSet,
		       double score) {
  scores[iSet.getPTId()] = score;
}


void PreTree::consume(Train* train,
		      Forest* forest,
		      Leaf* leaf) const {
  train->consumeInfo(infoLocal);
  
  forest->consumeTree(nodeVec, scores);
  forest->consumeBits(splitBits, observedBits, bitEnd);

  leaf->consumeTerminals(this, terminalMap);
}


void PreTree::setTerminals(SampleMap smTerminal) {
  terminalMap = std::move(smTerminal);

  leafMerge();
  setLeafIndices();
}


void PreTree::setLeafIndices() {
  vector<IndexRange> dom = Forest::leafDominators(nodeVec);
  for (auto ptIdx : terminalMap.ptIdx) {
    nodeVec[ptIdx].setLeaf(dom[ptIdx].getStart());
  }
}


void PreTree::leafMerge() {
  if (leafMax == 0 || leafCount <= leafMax) {
    return;
  }

  IndexT excessLeaves = leafCount - leafMax;
  IndexT height = getHeight();

  // Assigns parent indices and initializes information.
  vector<IndexT> ptParent(height);
  vector<PTMerge> mergeNode(height);
  for (IndexT ptId = 0; ptId < height; ptId++) {
    mergeNode[ptId].ptId = ptId;
    mergeNode[ptId].infoDom = infoNode[ptId];
    if (isNonterminal(ptId)) {
      IndexT kidLeft = ptId + getDelIdx(ptId);
      ptParent[kidLeft] = ptId;
      ptParent[kidLeft + 1] = ptId;
    }
  }

  // Accumulates sum of dominated info values.
  for (IndexT ptId = height - 1; ptId > 0; ptId--) {
    IndexT idParent = ptParent[ptId];
    mergeNode[idParent].infoDom += mergeNode[ptId].infoDom;
  }

  // Heap orders nonterminals by 'infoDom' value.
  priority_queue<PTMerge, vector<PTMerge>, InfoCompare> infoQueue;
  for (IndexT ptId = 0; ptId < height; ptId++) {
    if (isNonterminal(ptId)) {
      ////cout << "Inserting " << mergeNode[ptId].ptId << endl;
      infoQueue.emplace(mergeNode[ptId]);
    }
  }
  //cout << infoQueue.size() << " nonterminals in queue + " << leafCount << " leaves out of " << height << endl;
  
  vector<IndexT> ptMerged(height);
  iota(ptMerged.begin(), ptMerged.end(), 0);

  // Pops nonterminals in increasing 'infoDom' order.
  // 'infoDom' value is monotone increasing ascending a subtree, so
  // offspring always popped before dominator.

  BV mergedTerminal(height);
  IndexT nMerged = 0;
  while (nMerged < excessLeaves) {
    PTMerge ntMerged = infoQueue.top();
    infoQueue.pop();
    IndexT idMerged = ntMerged.ptId;
    mergedTerminal.setBit(idMerged);

    // Both offspring should be either leaf or merged.
    IndexT idKid = idMerged + getDelIdx(idMerged);
    ptMerged[idKid] = idMerged;
    ptMerged[idKid+1] = idMerged;
    nMerged++;
  }

  // Copies unmerged nodes into new node vector.
  vector<DecNode> nvFinal;
  vector<double> scoresFinal;
  vector<IndexT> old2New(height);
  fill(old2New.begin(), old2New.end(), height); // Inattainable index.
  for (IndexT ptId = 0; ptId < height; ptId++) {
    if (ptMerged[ptId] == ptId) { // Not merged away.
      old2New[ptId] = nvFinal.size();
      nvFinal.emplace_back(nodeVec[ptId]);
      scoresFinal.emplace_back(scores[ptId]);
    }
  }

  // Resets delIdx to reflect new indices.
  for (IndexT ptId = 0; ptId < height; ptId++) {
    if (old2New[ptId] == height) // Merged away.
      continue;
    IndexT ptIdNew = old2New[ptId];
    if (mergedTerminal.testBit(ptId)) {
      nvFinal[ptIdNew].resetTerminal();
    }
    else {
      IndexT kidL = getDelIdx(ptId) + ptId;
      nvFinal[ptIdNew].resetDelIdx(old2New[kidL] - ptIdNew);
    }
  }

  // Passes dominating merged node downward.
  for (IndexT ptId = 0; ptId < height; ptId++) {
    IndexT targ = ptMerged[ptId];
    if (targ != ptId)
      ptMerged[ptId] = ptMerged[targ];
  }

  // Resets terminal node indices.
  vector<vector<IndexT>> rangeMerge(nvFinal.size()); // Wasteful.
  for (IndexT rangeIdx = 0; rangeIdx < terminalMap.range.size(); rangeIdx++) {
    IndexT ptId = terminalMap.ptIdx[rangeIdx];
    IndexT termMerged = old2New[ptMerged[ptId]];
    rangeMerge[termMerged].push_back(rangeIdx);
  }

  // Rebuilds terminal map using merged ranges.
  SampleMap tmFinal;
  for (IndexT ptId = 0; ptId < rangeMerge.size(); ptId++) {
    if (rangeMerge[ptId].empty())
      continue;
    tmFinal.ptIdx.push_back(ptId);
    IndexT idxStart = tmFinal.sampleIndex.size();
    for (IndexT rangeIdx : rangeMerge[ptId]) {
      IndexRange& range = terminalMap.range[rangeIdx];
      for (IndexT idx = range.getStart(); idx != range.getEnd(); idx++) {
	IndexT stIdx = terminalMap.sampleIndex[idx];
	tmFinal.sampleIndex.push_back(stIdx);
      }
    }
    tmFinal.range.emplace_back(idxStart, tmFinal.sampleIndex.size() - idxStart);
  }

  nodeVec = nvFinal;
  scores = scoresFinal;
  terminalMap = tmFinal;
}


IndexT PreTree::checkFrontier(const vector<IndexT>& stMap) const {
  vector<bool> ptSeen(getHeight());
  IndexT nonLeaf = 0;
  for (auto ptIdx : stMap) {
    if (!ptSeen[ptIdx]) {
      if (isNonterminal(ptIdx)) {
	nonLeaf++;
      }
      ptSeen[ptIdx] = true;
    }
  }

  return nonLeaf;
}
