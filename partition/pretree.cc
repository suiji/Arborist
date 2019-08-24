// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file pretree.cc

   @brief Methods implementing production and consumption of the pre-tree.

   @author Mark Seligman

 */

#include "bv.h"
#include "pretree.h"
#include "ptnode.h"
#include "decnode.h"
#include "frontier.h"
#include "forest.h"
#include "summaryframe.h"
#include "callback.h"

#include <queue>


// Records terminal-node information for elements of the next level in the pre-tree.
// A terminal node may later be converted to non-terminal if found to be splitable.
// Initializing as terminal by default offers several advantages, such as avoiding
// the need to revise dangling non-terminals from an earlier level.
//

size_t PreTree::heightEst = 0;
size_t PreTree::leafMax = 0;


/**
   @brief Caches the row count and computes an initial estimate of node count.

   @param _nSamp is the number of samples.

   @param _minH is the minimal splitable index node size.

   @param leafMax is a user-specified limit on the number of leaves.
 */
void PreTree::immutables(size_t _nSamp, size_t _minH, size_t leafMax) {
  // Static initial estimate of pre-tree heights employs a minimal enclosing
  // balanced tree.  This is probably naive, given that decision trees
  // are not generally balanced.
  //
  // In any case, 'heightEst' is re-estimated following construction of the
  // first PreTree block.  Hence the value is not really immutable.  Nodes
  // can also be reallocated during the interlevel pass as needed.
  //
  size_t twoL = 1; // 2^level, beginning from level zero (root).
  while (twoL * _minH < _nSamp) {
    twoL <<= 1;
  }

  // Terminals plus accumulated nonterminals.
  heightEst = (twoL << 2); // - 1, for exact count.

  PreTree::leafMax = leafMax;
}


void PreTree::deImmutables() {
  leafMax = heightEst = 0;
}


/**
   @brief Per-tree initializations.

   @param _treeBlock is the number of PreTree initialize.

   @return void.
 */

PreTree::PreTree(const SummaryFrame* frame,
                 const Frontier* frontier) :
  bagCount(frontier->getBagCount()),
  height(1),
  leafCount(1),
  bitEnd(0),
  nodeVec(vector<PTNode>(2*bagCount - 1)), // Maximum possible nodes.
  splitBits(new BV(heightEst * frame->getCardExtent())) { // Initial estimate.
}


/**
   @brief Per-tree finalizer.
 */
PreTree::~PreTree() {
  delete splitBits;
}


void PreTree::setLeft(const IndexSet* iSet, IndexT pos) {
  splitBits->setBit(pos + nodeVec[iSet->getPTId()].getBitOffset(splitCrit));
}


IndexT PTNode::getBitOffset(const vector<SplitCrit>& splitCrit) const {
  return splitCrit[critOffset].getBitOffset();
}
    
    
void PreTree::reserve(size_t height) {
  while (heightEst <= height) // Assigns next power-of-two above 'height'.
    heightEst <<= 1;
}


void PreTree::nonterminal(double info, IndexSet* iSet) {
  nodeVec[iSet->getPTId()].nonterminal(info, height - iSet->getPTId(), splitCrit.size());
  terminalOffspring();
}


void PreTree::critBits(const IndexSet* iSet, unsigned int predIdx, unsigned int cardinality) {
  nodeVec[iSet->getPTId()].bumpCriterion();
  splitCrit.emplace_back(predIdx, bitEnd);
  bitEnd += cardinality;
  splitBits = splitBits->Resize(bitEnd);
}


void PreTree::critCut(const IndexSet* iSet, unsigned int predIdx, const IndexRange& rankRange) {
  nodeVec[iSet->getPTId()].bumpCriterion();
  splitCrit.emplace_back(predIdx, rankRange);
}


const vector<unsigned int> PreTree::consume(ForestTrain* forest, unsigned int tIdx, vector<double>& predInfo) {
  height = LeafMerge();
  forest->treeInit(tIdx, height);
  consumeNonterminal(forest, predInfo);
  forest->appendBits(splitBits, bitEnd, tIdx);

  return frontierConsume(forest);
}


void PreTree::consumeNonterminal(ForestTrain *forest, vector<double> &predInfo) const {
  fill(predInfo.begin(), predInfo.end(), 0.0);
  for (IndexT idx = 0; idx < height; idx++) {
    nodeVec[idx].consumeNonterminal(forest, predInfo, idx, splitCrit);
  }
}


void PTNode::consumeNonterminal(ForestTrain* forest, vector<double>& predInfo, unsigned int idx, const vector<SplitCrit>& splitCrit) const {
  if (isNonTerminal()) {
    SplitCrit crit(splitCrit[critOffset]);
    forest->nonTerminal(idx, lhDel, crit);
    predInfo[crit.predIdx] += info;
  }
}


void PreTree::subtreeFrontier(const vector<unsigned int>& stTerm) {
  for (auto & stIdx : stTerm) {
    termST.push_back(stIdx);
  }
}


const vector<unsigned int> PreTree::frontierConsume(ForestTrain* forest) const {
  vector<unsigned int> frontierMap(termST.size());
  vector<unsigned int> pt2Leaf(height);
  fill(pt2Leaf.begin(), pt2Leaf.end(), height); // Inattainable leaf index.

  unsigned int leafIdx = 0;
  unsigned int stIdx = 0;
  for (auto ptIdx : termST) {
    if (pt2Leaf[ptIdx] == height) {
      forest->terminal(ptIdx, leafIdx);
      pt2Leaf[ptIdx] = leafIdx++;
    }
    frontierMap[stIdx++] = pt2Leaf[ptIdx];
  }

  return frontierMap;
}


unsigned int PreTree::getBitWidth() {
  return BV::slotAlign(bitEnd);
}


/**
   @brief Workspace for merging PTNodes:  copies 'info' and records
   offsets and merge state.
 */
class PTMerge {
public:
  FltVal info;
  unsigned int ptId;
  unsigned int idMerged;
  unsigned int root;
  unsigned int parId;
  unsigned int idSib; // Sibling id, if not root else zero.
  bool descLH; // Whether this is left descendant of some node.
};


/**
   @brief Information-base comparator for queue ordering.
*/
class InfoCompare {
public:
  bool operator() (const PTMerge &a , const PTMerge &b) {
    return a.info > b.info;
  }
};


unsigned int PreTree::LeafMerge() {
  if (leafMax == 0 || leafCount <= leafMax) {
    return height;
  }

  vector<PTMerge> ptMerge(height);
  priority_queue<PTMerge, vector<PTMerge>, InfoCompare> infoQueue;

  auto leafProb = CallBack::rUnif(height);
  ptMerge[0].parId = 0;
  unsigned int ptId = 0;
  for (auto & merge : ptMerge) {
    merge.info = leafProb[ptId];
    merge.ptId = ptId;
    merge.idMerged = height;
    merge.root = height; // Merged away iff != height.
    merge.descLH = ptId != 0 && getLHId(merge.parId) == ptId;
    merge.idSib = ptId == 0 ? 0 : (merge.descLH ? getRHId(merge.parId) : getLHId(merge.parId));
    if (isNonTerminal(ptId)) {
      ptMerge[getLHId(ptId)].parId = ptMerge[getRHId(ptId)].parId = ptId;
      if (isMergeable(ptId)) {
        infoQueue.push(merge);
      }
    }
    ptId++;
  }

  // Merges and pops mergeable nodes and pushes newly mergeable parents.
  //
  unsigned int leafDiff = leafCount - leafMax;
  while (leafDiff-- > 0) {
    unsigned int ptTop = infoQueue.top().ptId;
    infoQueue.pop();
    ptMerge[ptTop].root = ptTop;
    unsigned int parId = ptMerge[ptTop].parId;
    unsigned int idSib = ptMerge[ptTop].idSib;
    if ((!isNonTerminal(idSib) || ptMerge[idSib].root != height)) {
      infoQueue.push(ptMerge[parId]);
    }
  }

  // Pushes down roots.  Roots remain in node list, but descendants
  // merged away.
  //
  unsigned int heightMerged = 0;
  for (unsigned int ptId = 0; ptId < height; ptId++) {
    unsigned int root = ptMerge[ptId].root;
    if (root != height && isNonTerminal(ptId)) {
      ptMerge[getLHId(ptId)].root = ptMerge[getRHId(ptId)].root = root;
    }
    if (root == height || root == ptId) { // Unmerged or root:  retained.
      nodeVec[ptId].setTerminal(); // Will reset if encountered as parent.
      if (ptMerge[ptId].descLH) {
        unsigned int parId = ptMerge[ptId].parId;
        nodeVec[parId].setNonterminal(heightMerged - ptMerge[parId].idMerged);
      }
      ptMerge[ptId].idMerged = heightMerged++;
    }
  }

  // Packs nodeVec[] with retained nodes.
  //
  for (unsigned int ptId = 0; ptId < height; ptId++) {
    unsigned int idMerged = ptMerge[ptId].idMerged;
    if (idMerged != height) {
      nodeVec[idMerged] = nodeVec[ptId];
    }
  }

  // Remaps frontier to merged terminals.
  //
  for (auto & ptId : termST) {
    unsigned int root = ptMerge[ptId].root;
    ptId = ptMerge[(root == height) ? ptId : root].idMerged;
  }

  return heightMerged;
}
