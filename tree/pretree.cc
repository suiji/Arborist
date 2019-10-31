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
#include "crit.h"
#include "indexset.h"

#include "callback.h"
#include <queue>
#include <vector>
#include <algorithm>


IndexT PreTree::heightEst = 0;
IndexT PreTree::leafMax = 0;


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


  /**
     @return BV-aligned length of used portion of split vector.
  */

size_t PreTree::getBitWidth(){
    return BV::slotAlign(bitEnd);
  }


PreTree::PreTree(PredictorT cardExtent,
	  IndexT bagCount_) :
    bagCount(bagCount_),
    height(1),
    leafCount(1),
    bitEnd(0),
    nodeVec(vector<PTNode<DecNode>>(2*bagCount - 1)), // Maximum possible nodes
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


  /**
     @brief Refines the height estimate using the actual height of a
     constructed PreTree.

     @param height is an actual height value.
  */
void PreTree::reserve(IndexT height) {
    while (heightEst <= height) // Assigns next power-of-two above 'height'.
      heightEst <<= 1;
  }

  /**
     @brief Dispatches nonterminal method according to split type.
   */
void PreTree::nonterminal(double info,
                   class IndexSet* iSet) {
    nodeVec[iSet->getPTId()].nonterminal(info, height - iSet->getPTId(), crit.size());
    terminalOffspring();
  }



  
  /**
     @brief Appends criterion for bit-based branch.

     @param predIdx is the criterion predictor.

     @param cardinality is the predictor's cardinality.
  */
void PreTree::critBits(const class IndexSet* iSet,
                PredictorT predIdx,
                PredictorT cardinality) {
    nodeVec[iSet->getPTId()].bumpCriterion();
    crit.emplace_back(predIdx, bitEnd);
    bitEnd += cardinality;
    splitBits = splitBits->Resize(bitEnd);
  }



  /**
     @brief Appends criterion for cut-based branch.
     
     @param rankRange bounds the cut-defining ranks.
  */
void PreTree::critCut(const class IndexSet* iSet,
               PredictorT predIdx,
	       double quantRank) {
    nodeVec[iSet->getPTId()].bumpCriterion();
    crit.emplace_back(predIdx, quantRank);
  }


const vector<IndexT> PreTree::consume(ForestCresc<DecNode> *forest,
                                     unsigned int tIdx,
                                     vector<double> &predInfo) {
    forest->treeInit(tIdx, height);
    consumeNonterminal(forest, predInfo);
    forest->appendBits(splitBits, bitEnd, tIdx);

    return frontierConsume(forest);
  }


void PreTree::consumeNonterminal(ForestCresc<DecNode> *forest,
                          vector<double> &predInfo)  {
    fill(predInfo.begin(), predInfo.end(), 0.0);
    for (IndexT idx = 0; idx < height; idx++) {
      nodeVec[idx].consumeNonterminal(forest, predInfo, idx, crit);
    }
  }

void PreTree::setLeft(const class IndexSet* iSet,
               IndexT pos) {
      splitBits->setBit(pos + nodeVec[iSet->getPTId()].getBitOffset(crit));
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
      if (root != height && isNonTerminal(ptId)) {
	ptMerge[getLHId(ptId)].root = ptMerge[getRHId(ptId)].root = root;
      }
      if (root == height || root == ptId) { // Unmerged or root:  retained.
	nodeVec[ptId].setTerminal(); // Will reset if encountered as parent.
	if (ptMerge[ptId].descLH) {
	  IndexT parId = ptMerge[ptId].parId;
	  nodeVec[parId].setNonterminal(heightMerged - ptMerge[parId].idMerged);
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

  
  /**
     @brief Absorbs the terminal list and merges, if requested.

     Side-effects the frontier map.

     @param stTerm are subtree-relative indices.  These must be mapped to
     sample indices if the subtree is proper.
  */
void PreTree::finish(const vector<IndexT>& stTerm) {
    for (auto & stIdx : stTerm) {
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
    merge.descLH = ptId != 0 && preTree->getLHId(merge.parId) == ptId;
    merge.idSib = ptId == 0 ? 0 : (merge.descLH ? preTree->getRHId(merge.parId) : preTree->getLHId(merge.parId));
    if (preTree->isNonTerminal(ptId)) {
      ptMerge[preTree->getLHId(ptId)].parId = ptMerge[preTree->getRHId(ptId)].parId = ptId;
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
    if ((!preTree->isNonTerminal(idSib) || ptMerge[idSib].root != height)) {
      infoQueue.push(ptMerge[parId]);
    }
  }
  return ptMerge;
  }
