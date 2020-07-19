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

#include "callback.h"
#include <queue>
#include <vector>


IndexT PreTree::heightEst = 0;
IndexT PreTree::leafMax = 0;


size_t PreTree::getBitWidth(){
    return BV::slotAlign(bitEnd);
  }


PreTree::PreTree(PredictorT cardExtent,
	  IndexT bagCount_) :
    bagCount(bagCount_),
    height(1),
    leafCount(1),
    bitEnd(0),
    nodeVec(vector<PTNode>(2*bagCount - 1)), // Maximum possible nodes
    splitBits(new BV(heightEst * cardExtent)) { // Initial estimate.
}


PreTree::~PreTree() {
  delete splitBits;
}


void PreTree::immutables(IndexT nSamp, IndexT minH, IndexT leafMax_) {
  // Static initial estimate of pre-tree heights employs a minimal enclosing
  // balanced tree.  This is probably naive, given that decision trees
  // are not generally balanced.
  //
  // In any case, 'heightEst' is re-estimated following construction of the
  // first PreTree block.  Hence the value is not really immutable.  Nodes
  // can also be reallocated during the interlevel pass as needed.
  //
    IndexT twoL = 1; // 2^level, beginning from level zero (root).
    while (twoL * minH < nSamp) {
      twoL <<= 1;
    }

    // Terminals plus accumulated nonterminals.
    heightEst = (twoL << 2); // - 1, for exact count.

    leafMax = leafMax_;
  }



void PreTree::deImmutables() {
  leafMax = heightEst = 0;
}


void PreTree::reserve(IndexT height) {
  while (heightEst <= height) // Assigns next power-of-two above 'height'.
    heightEst <<= 1;
}


void PreTree::nonterminalInc(const SplitNux& nux) {
  PTNode::setNonterminal(nodeVec, nux, height);
}


void PreTree::nonterminal(const SplitNux& nux) {
  offspring(1);
  PTNode::setNonterminal(nodeVec, nux, height);
}


void PTNode::setNonterminal(vector<PTNode>& nodeVec, const SplitNux& nux, IndexT height) {
  nodeVec[nux.getPTId()].setNonterminal(nux, height);
}


void PTNode::setNonterminal(const SplitNux& nux,
                            IndexT height) {
  setDelIdx(height - 2 - nux.getPTId());
  info = nux.getInfo();
}


void PreTree::critBits(const SplitNux* nux,
		       PredictorT cardinality,
		       const vector<PredictorT> bitsTrue) {
  nodeVec[nux->getPTId()].critBits(nux, bitEnd);
  splitBits = splitBits->Resize(bitEnd + cardinality);
  for (auto bit : bitsTrue) {
    splitBits->setBit(bitEnd + bit);
  }
  bitEnd += cardinality;
}


void PreTree::critCut(const SplitNux* nux, const class SplitFrontier* splitFrontier) {
  nodeVec[nux->getPTId()].critCut(nux, splitFrontier);
}


const vector<IndexT> PreTree::consume(ForestCresc<DecNode>* forest,
                                     unsigned int tIdx,
                                     vector<double>& predInfo) {
  forest->treeInit(tIdx, height);
  consumeNonterminal(forest, predInfo);
  forest->appendBits(splitBits, bitEnd, tIdx);

  return frontierConsume(forest);
}


const vector<IndexT> PreTree::frontierConsume(ForestCresc<DecNode> *forest) const {
  vector<IndexT> frontierMap(termST.size());
  vector<IndexT> pt2Leaf(height);
  fill(pt2Leaf.begin(), pt2Leaf.end(), height); // Inattainable leaf index.

  IndexT leafIdx = 0;
  IndexT stIdx = 0;
  for (auto ptIdx : termST) {
    if (pt2Leaf[ptIdx] == height) {
      forest->terminal(ptIdx, leafIdx);
      pt2Leaf[ptIdx] = leafIdx++;
    }
    frontierMap[stIdx++] = pt2Leaf[ptIdx];
  }

  return frontierMap;
}


void PreTree::consumeNonterminal(ForestCresc<DecNode>* forest,
				 vector<double>& predInfo)  {
  fill(predInfo.begin(), predInfo.end(), 0.0);
  for (IndexT idx = 0; idx < height; idx++) {
    nodeVec[idx].consumeNonterminal(forest, predInfo, idx);
  }
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
  for (auto & ptId : termST) {
    IndexT root = ptMerge[ptId].root;
    ptId = ptMerge[(root == height) ? ptId : root].idMerged;
  }

  return heightMerged;
}

  
void PreTree::finish(const vector<IndexT>& stTerm) {
  for (auto stIdx : stTerm) {
    termST.push_back(stIdx);
  }

  height = leafMerge();
}


void PreTree::blockBump(IndexT& _height,
                        IndexT& _maxHeight,
                        size_t& _bitWidth,
                        IndexT& _leafCount,
                        IndexT& _bagCount) {
  _height += height;
  _maxHeight = max(height, _maxHeight);
  _bitWidth += getBitWidth();
  _leafCount += leafCount;
  _bagCount += bagCount;
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
