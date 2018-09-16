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
#include "splitcand.h"
#include "forest.h"
#include "framemap.h"
#include "callback.h"

#include <queue>


// Records terminal-node information for elements of the next level in the pre-tree.
// A terminal node may later be converted to non-terminal if found to be splitable.
// Initializing as terminal by default offers several advantages, such as avoiding
// the need to revise dangling non-terminals from an earlier level.
//

unsigned int PreTree::heightEst = 0;
unsigned int PreTree::leafMax = 0;


/**
   @brief Caches the row count and computes an initial estimate of node count.

   @param _nSamp is the number of samples.

   @param _minH is the minimal splitable index node size.

   @return void.
 */
void PreTree::Immutables(unsigned int _nSamp, unsigned int _minH, unsigned int _leafMax) {
  // Static initial estimate of pre-tree heights employs a minimal enclosing
  // balanced tree.  This is probably naive, given that decision trees
  // are not generally balanced.
  //
  // In any case, 'heightEst' is re-estimated following construction of the
  // first PreTree block.  Hence the value is not really immutable.  Nodes
  // can also be reallocated during the interlevel pass as needed.
  //
  unsigned twoL = 1; // 2^level, beginning from level zero (root).
  while (twoL * _minH < _nSamp) {
    twoL <<= 1;
  }

  // Terminals plus accumulated nonterminals.
  heightEst = (twoL << 2); // - 1, for exact count.

  leafMax = _leafMax;
}


void PreTree::DeImmutables() {
  leafMax = heightEst = 0;
}


/**
   @brief Per-tree initializations.

   @param _treeBlock is the number of PreTree initialize.

   @return void.
 */
PreTree::PreTree(const FrameTrain *frameTrain_,
                 unsigned int bagCount_) :
  frameTrain(frameTrain_),
  bagCount(bagCount_),
  nodeCount(heightEst), // Initial estimate.
  height(1),
  leafCount(1),
  bitEnd(0) {
  nodeVec = new PTNode[nodeCount];
  splitBits = BitFactory();
}


/**
   @brief Per-tree finalizer.
 */
PreTree::~PreTree() {
  delete [] nodeVec;
  delete splitBits;
}


/**
   @brief Sets specified bit in splitting bit vector.

   @param id is the index node for which the LH bit is set.

   @param pos is the bit position beyond to set.

   @return void.
*/
void PreTree::LHBit(int idx, unsigned int pos) {
  splitBits->setBit(nodeVec[idx].splitVal.offset + pos);
}


/**
   @brief Refines the height estimate using the actual height of a
   constructed PreTree.

   @param height is an actual height value.

   @return void.
 */
void PreTree::Reserve(unsigned int height) {
  while (heightEst <= height) // Assigns next power-of-two above 'height'.
    heightEst <<= 1;
}


/**
   @brief Allocates a zero-valued bit string for the current (pre)tree.

   @return pointer to allocated vector, possibly zero-length.
*/
BV *PreTree::BitFactory() {
  // Should be wide enough to hold all factor bits for an entire tree:
  //    estimated #nodes * width of widest factor.
  return new BV(nodeCount * frameTrain->CardMax());
}


/**
   @brief Fills in some fields for (generic) node found splitable.

   @param _id is the node index.

   @param _info is the information content.

   @param _predIdx is the splitting predictor index.

   @return void.
*/
void PreTree::branchFac(const SplitCand& argMax, unsigned int id) {
  nodeVec[id].SplitFac(argMax.getPredIdx(), height - id, bitEnd, argMax.getInfo());
  TerminalOffspring();
  bitEnd += frameTrain->FacCard(argMax.getPredIdx());
}


void PreTree::branchNum(const SplitCand &argMax, unsigned int id) {
  nodeVec[id].splitNum(argMax, height - id);
  TerminalOffspring();
}


void PTNode::splitNum(const SplitCand &cand, unsigned int lhDel) {
  this->predIdx = cand.getPredIdx();
  this->lhDel = lhDel;
  this->splitVal.rankRange = cand.getRankRange();
  this->info = cand.getInfo();
}


/**
   @brief Ensures sufficient space to accomodate the next level for nodes
   just split.  If necessary, doubles existing vector sizes.

   N.B.:  reallocations incur considerable resynchronization costs if
   precipitated from the coprocessor.

   @param splitNext is the count of splits in the upcoming level.

   @param leafNext is the count of leaves in the upcoming level.

   @return current height;
*/
void PreTree::Level(unsigned int splitNext, unsigned int leafNext) {
  if (height + splitNext + leafNext > nodeCount) {
    ReNodes();
  }

  unsigned int bitMin = bitEnd + splitNext * frameTrain->CardMax();
  if (bitMin > 0) {
    splitBits = splitBits->Resize(bitMin);
  }
}


/**
 @brief Guesstimates safe height by doubling high watermark.

 @return void.
*/
void PreTree::ReNodes() {
  nodeCount <<= 1;
  PTNode *PTtemp = new PTNode[nodeCount];
  for (unsigned int i = 0; i < height; i++)
    PTtemp[i] = nodeVec[i];

  delete [] nodeVec;
  nodeVec = PTtemp;
}


const vector<unsigned int> PreTree::consume(ForestTrain *forest, unsigned int tIdx, vector<double> &predInfo) {
  height = LeafMerge();
  forest->initNode(height);
  consumeNonterminal(forest, tIdx, predInfo);
  forest->BitProduce(splitBits, bitEnd);
  forest->setHeights(tIdx); // height and bit lengths known.

  return FrontierConsume(forest, tIdx);
}


/**
   @brief Consumes nonterminal information into the dual-use vectors needed by the decision tree.  Leaf information is post-assigned by the response-dependent Sample methods.

   @param forest inputs/outputs the updated forest.

   @return void, with output reference parameter.
*/
void PreTree::consumeNonterminal(ForestTrain *forest, unsigned int tIdx, vector<double> &predInfo) const {
  fill(predInfo.begin(), predInfo.end(), 0.0);
  for (unsigned int idx = 0; idx < height; idx++) {
    nodeVec[idx].consumeNonterminal(frameTrain, forest, tIdx, predInfo, idx);
  }
}


/**
   @brief Consumes the node fields of nonterminals (splits).

   @param forest outputs the growing forest node vector.

   @param tIdx is the index of the tree being produced.

   @return void, with side-effected Forest.
 */
void PTNode::consumeNonterminal(const FrameTrain *frameTrain, ForestTrain *forest, unsigned int tIdx, vector<double> &predInfo, unsigned int idx) const {
  if (NonTerminal()) {
    forest->NonTerminal(frameTrain, tIdx, idx, this);
    predInfo[predIdx] += info;
  }
}


/**
   @brief Absorbs the terminal list from a completed subtree.

   @param stTerm are subtree-relative indices.  These must be mapped to
   sample indices if the subtree is proper.

   @return void, with side-effected frontier map.
 */
void PreTree::SubtreeFrontier(const vector<unsigned int> &stTerm) {
  for (auto & stIdx : stTerm) {
    termST.push_back(stIdx);
  }
}


/**
   @brief Constructs mapping from sample indices to leaf indices.

   @param tIdx is the index of the tree being produced.

   @return Reference to rewritten map, with side-effected Forest.
 */
const vector<unsigned int> PreTree::FrontierConsume(ForestTrain *forest, unsigned int tIdx) const {
  vector<unsigned int> frontierMap(termST.size());
  vector<unsigned int> pt2Leaf(height);
  fill(pt2Leaf.begin(), pt2Leaf.end(), height); // Inattainable leaf index.

  unsigned int leafIdx = 0;
  for (unsigned int stIdx = 0; stIdx < termST.size(); stIdx++) {
    unsigned int ptIdx = termST[stIdx];
    if (pt2Leaf[ptIdx] == height) {
      forest->LeafProduce(tIdx, ptIdx, leafIdx);
      pt2Leaf[ptIdx] = leafIdx++;
    }
    frontierMap[stIdx] = pt2Leaf[ptIdx];
  }

  return frontierMap;
}


/**
   @return BV-aligned length of used portion of split vector.
 */
unsigned int PreTree::BitWidth() {
  return BV::SlotAlign(bitEnd);
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


class InfoCompare {
public:
  bool operator() (const PTMerge *a , const PTMerge *b) {
    return a->info > b->info;
  }
};


unsigned int PreTree::LeafMerge() {
  if (leafMax == 0 || leafCount <= leafMax) {
    return height;
  }

  unsigned int leafDiff = leafCount - leafMax;
  vector<PTMerge> ptMerge(height);
  priority_queue<PTMerge*, vector<PTMerge*>, InfoCompare> infoQueue;
  auto leafProb = CallBack::rUnif(leafCount);

  ptMerge[0].parId = 0;
  for (unsigned int ptId = 0; ptId < height; ptId++) {
    PTMerge *merge = &ptMerge[ptId];
    merge->info = leafProb[ptId];
    merge->ptId = ptId;
    merge->idMerged = height;
    merge->root = height; // Merged away iff != height.
    merge->descLH = ptId != 0 && LHId(merge->parId) == ptId;
    merge->idSib = ptId == 0 ? 0 : (merge->descLH ? RHId(merge->parId) : LHId(merge->parId));
    if (NonTerminal(ptId)) {
      ptMerge[LHId(ptId)].parId = ptMerge[RHId(ptId)].parId = ptId;
      if (Mergeable(ptId)) {
        infoQueue.push(merge);
      }
    }
  }

  // Merges and pops mergeable nodes and pushes newly mergeable parents.
  //

  while (leafDiff-- > 0) {
    unsigned int ptTop = infoQueue.top()->ptId;
    infoQueue.pop();
    ptMerge[ptTop].root = ptTop;
    unsigned int parId = ptMerge[ptTop].parId;
    unsigned int idSib = ptMerge[ptTop].idSib;
    if ((!NonTerminal(idSib) || ptMerge[idSib].root != height)) {
      infoQueue.push(&ptMerge[parId]);
    }
  }

  // Pushes down roots.  Roots remain in node list, but descendants
  // merged away.
  //
  unsigned int heightMerged = 0;
  for (unsigned int ptId = 0; ptId < height; ptId++) {
    unsigned int root = ptMerge[ptId].root;
    if (root != height && NonTerminal(ptId)) {
      ptMerge[LHId(ptId)].root = ptMerge[RHId(ptId)].root = root;
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
