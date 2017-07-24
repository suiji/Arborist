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
#include "forest.h"
#include "predblock.h"

#include <queue>

//#include <iostream>
//using namespace std;


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
PreTree::PreTree(const PMTrain *_pmTrain, unsigned int _bagCount) : pmTrain(_pmTrain), height(1), leafCount(1), bitEnd(0), bagCount(_bagCount) {
  nodeCount = heightEst;   // Initial height estimate.
  nodeVec = new PTNode[nodeCount];
  nodeVec[0].id = 0; // Root.
  nodeVec[0].lhDel = 0; // Initializes as terminal.
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
  splitBits->SetBit(nodeVec[idx].splitVal.offset + pos);
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
  return new BV(nodeCount * pmTrain->CardMax());
}


/**
   @brief Speculatively sets the two offspring slots as terminal lh, rh and changes status of this from terminal to nonterminal.

   @param _parId is the pretree index of the parent.

   @param ptLH outputs the left-hand node index.

   @param ptRH outputs the right-hand node index.

   @return void, with output reference parameters.
*/
void PreTree::TerminalOffspring(unsigned int _parId) {
  unsigned int ptLH = height++;
  nodeVec[_parId].lhDel = ptLH - _parId;
  nodeVec[ptLH].id = ptLH;
  nodeVec[ptLH].lhDel = 0;

  unsigned int ptRH = height++;
  nodeVec[ptRH].id = ptRH;
  nodeVec[ptRH].lhDel = 0;

  // Two more leaves for offspring, one fewer for this.
  leafCount++;
}


/**
   @brief Fills in some fields for (generic) node found splitable.

   @param _id is the node index.

   @param _info is the information content.

   @param _predIdx is the splitting predictor index.

   @return void.
*/
void PreTree::NonTerminalFac(double _info, unsigned int _predIdx, unsigned int _id) {
  TerminalOffspring(_id);
  PTNode *ptS = &nodeVec[_id];
  ptS->predIdx = _predIdx;
  ptS->info = _info;
  ptS->splitVal.offset = bitEnd;
  bitEnd += pmTrain->FacCard(_predIdx);
}


/**
   @brief Finalizes numeric-valued nonterminal.

   @param _info is the splitting information content.

   @param _predIdx is the splitting predictor index.

   @param _id is the node index.

   @return void.
*/
void PreTree::NonTerminalNum(double _info, unsigned int _predIdx, RankRange _rankRange, unsigned int _id) {
  TerminalOffspring(_id);
  PTNode *ptS = &nodeVec[_id];
  ptS->predIdx = _predIdx;
  ptS->info = _info;
  ptS->splitVal.rankRange = _rankRange;
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

  unsigned int bitMin = bitEnd + splitNext * pmTrain->CardMax();
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


/**
  @brief Consumes all pretree nonterminal information into crescent decision forest.

  @param forest grows by producing nodes and splits consumed from pre-tree.

  @param tIdx is the index of the tree being consumed/produced.

  @param predInfo accumulates the information contribution of each predictor.

  @return leaf map from consumed frontier.
*/
const std::vector<unsigned int> PreTree::Consume(ForestTrain *forest, unsigned int tIdx, std::vector<double> &predInfo) {
  forest->Origins(tIdx);
  LeafMerge(forest);
  NonterminalConsume(forest, tIdx, predInfo);
  forest->BitProduce(splitBits, bitEnd);

  return FrontierConsume(forest, tIdx);
}


/**
   @brief Consumes nonterminal information into the dual-use vectors needed by the decision tree.  Leaf information is post-assigned by the response-dependent Sample methods.

   @param forest inputs/outputs the updated forest.

   @return void, with output reference parameter.
*/
void PreTree::NonterminalConsume(ForestTrain *forest, unsigned int tIdx, std::vector<double> &predInfo) const {
  std::fill(predInfo.begin(), predInfo.end(), 0.0);
  for (unsigned int idx = 0; idx < height; idx++) {
    nodeVec[idx].NonterminalConsume(pmTrain, forest, tIdx, predInfo);
  }
}


/**
   @brief Consumes the node fields of nonterminals (splits).

   @param forest outputs the growing forest node vector.

   @param tIdx is the index of the tree being produced.

   @return void, with side-effected Forest.
 */
void PTNode::NonterminalConsume(const PMTrain *pmTrain, ForestTrain *forest, unsigned int tIdx, std::vector<double> &predInfo) const {
  if (NonTerminal()) {
    if (pmTrain->IsFactor(predIdx)) {
      forest->OffsetProduce(tIdx, id, predIdx, lhDel, splitVal.offset);
    }
    else {
      forest->RankProduce(tIdx, id, predIdx, lhDel, splitVal.rankRange.rankLow, splitVal.rankRange.rankHigh);
    }
    predInfo[predIdx] += info;
  }
}


/**
   @brief Absorbs the terminal list from a completed subtree.

   @param stTerm are subtree-relative indices.  These must be mapped to
   sample indices if the subtree is proper.

   @return void, with side-effected frontier map.
 */
void PreTree::SubtreeFrontier(const std::vector<unsigned int> &stTerm) {
  for(auto & stIdx : stTerm) {
    termST.push_back(stIdx);
  }
}


/**
   @brief Constructs mapping from sample indices to leaf indices.

   @param tIdx is the index of the tree being produced.

   @return Reference to rewritten map, with side-effected Forest.
 */
const std::vector<unsigned int> PreTree::FrontierConsume(ForestTrain *forest, unsigned int tIdx) const {
  std::vector<unsigned int> frontierMap(termST.size());
  std::vector<unsigned int> pt2Leaf(height);
  std::fill(pt2Leaf.begin(), pt2Leaf.end(), height); // Inattainable leaf index.

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


class InfoCompare {
public:
  bool operator() (const PTNode &a , const PTNode &b) {
    return a.info > b.info;
  }
};


void PreTree::LeafMerge(ForestTrain *forest) {
  if (leafMax == 0 || leafCount <= leafMax) {
    forest->NodeInit(height);
    return;
  }
  unsigned int leafCt = leafCount;
  
  std::priority_queue<PTNode, std::vector<PTNode>, InfoCompare> infoQueue;
  std::vector<unsigned int> mergeRoot(height);
  std::fill(mergeRoot.begin(), mergeRoot.end(), height); // Inattainable value.
  std::vector<bool> parSeen(height);
  std::fill(parSeen.begin(), parSeen.end(), false);
  // Initializes parent indices and pushes mergeable frontier
  // nodes.
  std::vector<unsigned int> parId(height);
  parId[0] = 0;
  for (unsigned int ptId = 0; ptId < height; ptId++) {
    if (NonTerminal(ptId)) {
      parId[LHId(ptId)] = parId[RHId(ptId)] = ptId;
      if (Mergeable(ptId)) {
	infoQueue.push(nodeVec[ptId]);
      }
    }
  }

  // Merges and pops mergeable nodes and pushes newly mergeable parents.
  while (leafCt > leafMax) {
    PTNode ptTop = infoQueue.top();
    unsigned int topId = ptTop.id;
    mergeRoot[topId] = topId;
    infoQueue.pop();
    leafCt--;
    unsigned int parIndex = parId[topId];
    if ((!NonTerminal(LHId(parIndex)) || mergeRoot[LHId(parIndex)] != height) &&
	(!NonTerminal(RHId(parIndex)) || mergeRoot[RHId(parIndex)] != height)) {
      infoQueue.push(nodeVec[parIndex]);
    }
  }

  // Pushes down merged roots.  Roots remain in node list, but descendants
  // merged away.
  //
  unsigned int heightMerged = 0;
  for (unsigned int idx = 0; idx < height; idx++) {
    unsigned int root = mergeRoot[idx];
    if (root != height) { // Merged, possibly a root.
      mergeRoot[LHId(idx)] = mergeRoot[RHId(idx)] = root;
      nodeVec[idx].lhDel = 0; // Resets to terminal.
    }
    if (root == height || root == idx) {
      nodeVec[idx].id = heightMerged++;
      if (idx > 0) { // Relink to parent.
        unsigned int parIndex = parId[idx];
	if (!parSeen[parIndex]) { // First child seen is left.
	  nodeVec[parIndex].lhDel = nodeVec[idx].id - nodeVec[parIndex].id;
	  parSeen[parIndex] = true;
	}
      }
    }
  }

  // Remaps frontier to merged terminals.
  for (auto & ptId : termST) {
    unsigned int root = mergeRoot[ptId];
    ptId = nodeVec[(root == height) ? ptId : root].id;
  }

  forest->NodeInit(heightMerged);
}
