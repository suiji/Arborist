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
#include "bv.h"
#include "pretree.h"
#include "splitfrontier.h"

#include "callback.h"
#include <queue>
#include <vector>

IndexT PreTree::leafMax = 0;


PreTree::PreTree(PredictorT cardExtent,
		 IndexT bagCount) :
  height(1),
  leafCount(1),
  nodeVec(vector<PTNode>(2*bagCount - 1)), // Preallocates maximum.
  splitBits(BV(bagCount * cardExtent)), // Vague estimate.
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

  if (nux.isFactor(sf)) {
    critBits(sf, nux);
  }
  else {
    critCut(sf, nux);
  }

  offspring(preallocated ? 0 : 1);
  nodeVec[nux.getPTId()].setNonterminal(nux, height);
}


void PreTree::critBits(const SplitFrontier* sf,
		       const SplitNux& nux) {
  auto bitPos = bitEnd;
  bitEnd += sf->critBitCount(nux);
  splitBits.resize(bitEnd);
  for (auto bit : sf->getTrueBits(nux)) {
    splitBits.setBit(bitPos + bit);
  }
  nodeVec[nux.getPTId()].critBits(nux, bitPos);
}


void PreTree::critCut(const SplitFrontier* sf,
		      const SplitNux& nux) {
  nodeVec[nux.getPTId()].critCut(nux, sf);
}


void PTNode::setNonterminal(const SplitNux& nux,
                            IndexT height) {
  setDelIdx(height - 2 - nux.getPTId());
  info = nux.getInfo();
}


const vector<IndexT> PreTree::consume(Forest* forest,
				      vector<double>& predInfo) {
  forest->treeInit(height);
  height = leafMerge();
  consumeNodes(forest, predInfo);
  forest->appendBits(splitBits, bitEnd);

  return sample2Leaf();
}


void PreTree::consumeNodes(Forest* forest,
			   vector<double>& predInfo)  {
  IndexT leafIdx = 0;
  for (IndexT idx = 0; idx < height; idx++) {
    nodeVec[idx].consume(forest, predInfo, idx, leafIdx);
  }
}


const vector<IndexT> PreTree::sample2Leaf() const {
  vector<IndexT> terminalMap(sampleMap.size()); // bagCount
  IndexT stIdx = 0;
  for (auto ptIdx : sampleMap) { // predIdx value of terminal is leaf index.
    terminalMap[stIdx++] = nodeVec[ptIdx].getPredIdx();
  }

  return terminalMap;
}


IndexT PreTree::checkFrontier(const vector<IndexT>& stMap) const {
  vector<bool> ptSeen(height);
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
  if (leafMax == 0 || leafCount <= leafMax) {
    return height;
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
  for (auto & ptId : sampleMap) {
    IndexT root = ptMerge[ptId].root;
    ptId = ptMerge[(root == height) ? ptId : root].idMerged;
  }

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
