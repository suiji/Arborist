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
#include "trainframe.h"
#include "indexset.h"
#include "leaf.h"
#include "samplemap.h"
#include "bv.h"
#include "splitfrontier.h"
#include "pretree.h"

#include "callback.h"
#include <queue>
#include <vector>

IndexT PreTree::leafMax = 0;


PreTree::PreTree(const TrainFrame* frame,
		 IndexT bagCount) :
  leafCount(0),
  infoLocal(vector<double>(frame->getNPred())),
  splitBits(BV(bagCount * frame->getCardExtent())), // Vague estimate.
  bitEnd(0) {
  offspring(1);
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

  if (nux.isFactor(sf)) {
    critBits(sf, nux);
  }
  else {
    critCut(sf, nux);
  }

  offspring(preallocated ? 0 : 1);
  DecNode& node = getNode(nux.getPTId());
  node.setInvert(nux.invertTest());
  node.setDelIdx(getHeight() - 2 - nux.getPTId());
  infoLocal[node.getPredIdx()] += nux.getInfo();
}


void PreTree::critBits(const SplitFrontier* sf,
		       const SplitNux& nux) {
  auto bitPos = bitEnd;
  splitBits.resize(exchange(bitEnd, bitEnd + sf->critBitCount(nux)));
  sf->setTrueBits(nux, &splitBits, bitPos);
  getNode(nux.getPTId()).critBits(&nux, bitPos);
}


void PreTree::critCut(const SplitFrontier* sf,
		      const SplitNux& nux) {
  getNode(nux.getPTId()).critCut(&nux, sf);
}


void PreTree::setScore(const SplitFrontier* splitFrontier,
		       const IndexSet& iSet) {
  scores[iSet.getPTId()] = splitFrontier->getScore(iSet);
}


void PreTree::consume(Train* train,
		      Forest* forest,
		      Leaf* leaf) const {
  train->consumeInfo(infoLocal);
  
  forest->consumeTree(nodeVec, scores);
  forest->consumeBits(splitBits, bitEnd);

  leaf->consumeTerminals(this, terminalMap);
}


void PreTree::setTerminals(SampleMap smTerminal) {
  terminalMap = move(smTerminal);

  (void) leafMerge();
  setLeafIndices();
}


void PreTree::setLeafIndices() {
  vector<IndexRange> dom = Forest::leafDominators(nodeVec);
  for (auto ptIdx : terminalMap.ptIdx) {
    nodeVec[ptIdx].setLeaf(dom[ptIdx].getStart());
  }
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


IndexT PreTree::leafMerge() {
  IndexT height = getHeight();
  if (leafMax == 0 || leafCount <= leafMax) {
    return height;
  }

  vector<IndexT> st2pt(terminalMap.indices.size()); // bagCount.
  IndexT rangeIdx = 0;
  for (auto range : terminalMap.range) {
    IndexT ptIdx = terminalMap.ptIdx[rangeIdx];
    for (IndexT idx = range.getStart(); idx != range.getEnd(); idx++) {
      IndexT stIdx = terminalMap.indices[idx];
      st2pt[stIdx] = ptIdx;
    }
    rangeIdx++;
  }

  vector<PTMerge<DecNode>> ptMerge = PTMerge<DecNode>::merge(this, height, leafCount - leafMax);

  // Pushes down roots.  Roots remain in node list, but descendants
  // merged away.
  //
  IndexT heightMerged = 0;
  for (IndexT ptId = 0; ptId < height; ptId++) {
    IndexT root = ptMerge[ptId].root;
    if (root != height && isNonterminal(ptId)) {
      ptMerge[getIdFalse(ptId)].root = ptMerge[getIdFalse(ptId)].root = root;
    }
    if (root == height || root == ptId) { // Unmerged or root:  retained.
      nodeVec[ptId].setTerminal(); // Will reset if encountered as parent.
      if (ptMerge[ptId].descTrue) {
	IndexT parId = ptMerge[ptId].parId;
	nodeVec[parId].setDelIdx(heightMerged - ptMerge[parId].idMerged);
      }
      ptMerge[ptId].idMerged = heightMerged++;
    }
  }
    
  // Packs nodeVec[] with retained nodes.
  //
  for (IndexT ptId = 0; ptId < height; ptId++) {
    IndexT idMerged = ptMerge[ptId].idMerged;
    if (idMerged != height) {
      nodeVec[idMerged] = nodeVec[ptId];
    }
  }

  // Remaps frontier to merged terminals.
  //
  for (auto & ptId : st2pt) {
    IndexT root = ptMerge[ptId].root;
    ptId = ptMerge[(root == height) ? ptId : root].idMerged;
  }

  // TODO:  Reform node/score and retype return value to void.
  return heightMerged;
}


template<typename nodeType>
vector<PTMerge<nodeType>> PTMerge<nodeType>::merge(const PreTree* preTree,
						   IndexT height,
						   IndexT leafDiff) {
  vector<PTMerge<nodeType>> ptMerge(height);
  priority_queue<PTMerge<nodeType>, vector<PTMerge<nodeType>>, InfoCompare<nodeType>> infoQueue;

  auto leafProb = CallBack::rUnif(height);
  ptMerge[0].parId = 0;
  IndexT ptId = 0;
  for (auto & merge : ptMerge) {
    merge.info = leafProb[ptId];
    merge.ptId = ptId;
    merge.idMerged = height;
    merge.root = height; // Merged away iff != height.
    merge.descTrue = ptId != 0 && preTree->getIdFalse(merge.parId) == ptId;
    merge.idSib = ptId == 0 ? 0 : (merge.descTrue ? preTree->getIdFalse(merge.parId) : preTree->getIdFalse(merge.parId));
    if (preTree->isNonterminal(ptId)) {
      ptMerge[preTree->getIdFalse(ptId)].parId = ptMerge[preTree->getIdFalse(ptId)].parId = ptId;
      if (preTree->isMergeable(ptId)) {
        infoQueue.push(merge);
      }
    }
    ptId++;
  }

  // Merges and pops mergeable nodes and pushes newly mergeable parents.
  //
  while (leafDiff-- > 0) {
    IndexT ptTop = infoQueue.top().ptId;
    infoQueue.pop();
    ptMerge[ptTop].root = ptTop;
    IndexT parId = ptMerge[ptTop].parId;
    IndexT idSib = ptMerge[ptTop].idSib;
    if ((!preTree->isNonterminal(idSib) || ptMerge[idSib].root != height)) {
      infoQueue.push(ptMerge[parId]);
    }
  }

  return ptMerge;
}
