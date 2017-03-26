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

//#include <iostream>
//using namespace std;


// Records terminal-node information for elements of the next level in the pre-tree.
// A terminal node may later be converted to non-terminal if found to be splitable.
// Initializing as terminal by default offers several advantages, such as avoiding
// the need to revise dangling non-terminals from an earlier level.
//

unsigned int PreTree::heightEst = 0;

/**
   @brief Caches the row count and computes an initial estimate of node count.

   @param _nSamp is the number of samples.

   @param _minH is the minimal splitable index node size.

   @return void.
 */
void PreTree::Immutables(unsigned int _nSamp, unsigned int _minH) {
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
}


void PreTree::DeImmutables() {
  heightEst = 0;
}


/**
   @brief Per-tree initializations.

   @param _treeBlock is the number of PreTree initialize.

   @return void.
 */
PreTree::PreTree(const PMTrain *_pmTrain, unsigned int _bagCount) : pmTrain(_pmTrain), height(1), leafCount(1), bitEnd(0), bagCount(_bagCount), info(std::vector<double>(pmTrain->NPred())) {
  std::fill(info.begin(), info.end(), 0.0);

  nodeCount = heightEst;   // Initial height estimate.
  nodeVec = new PTNode[nodeCount];
  nodeVec[0].id = 0; // Root.
  nodeVec[0].lhId = 0; // Initializes as terminal.
  splitBits = BitFactory();
}


/**
   @brief Per-tree finalizer.
 */
PreTree::~PreTree() {
  delete [] nodeVec;
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
  nodeVec[_parId].lhId = ptLH;
  nodeVec[ptLH].id = ptLH;
  nodeVec[ptLH].lhId = 0;

  unsigned int ptRH = height++;
  nodeVec[ptRH].id = ptRH;
  nodeVec[ptRH].lhId = 0;

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
  info[_predIdx] += _info;

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
  info[_predIdx] += _info;
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

  @return void, with side-effected forest.
*/
const std::vector<unsigned int> PreTree::DecTree(ForestTrain *forest, unsigned int tIdx, std::vector<double> &predInfo) {
  forest->Origins(tIdx);
  forest->NodeInit(height);
  NodeConsume(forest, tIdx);
  forest->BitProduce(splitBits, bitEnd);
  delete splitBits;

  for (unsigned int i = 0; i < info.size(); i++)
    predInfo[i] += info[i];

  return FrontierToLeaf(forest, tIdx);
}


/**
   @brief Consumes nonterminal information into the dual-use vectors needed by the decision tree.  Leaf information is post-assigned by the response-dependent Sample methods.

   @param forest inputs/outputs the updated forest.

   @return void, with output reference parameter.
*/
void PreTree::NodeConsume(ForestTrain *forest, unsigned int tIdx) {
  for (unsigned int idx = 0; idx < height; idx++) {
    nodeVec[idx].Consume(pmTrain, forest, tIdx);
  }
}


/**
   @brief Consumes the node fields of nonterminals (splits).

   @param forest outputs the growing forest node vector.

   @param tIdx is the index of the tree being produced.

   @return void, with side-effected Forest.
 */
void PTNode::Consume(const PMTrain *pmTrain, ForestTrain *forest, unsigned int tIdx) {
  if (lhId > 0) { // i.e., nonterminal
    if (pmTrain->IsFactor(predIdx)) {
      forest->OffsetProduce(tIdx, id, predIdx, lhId - id, splitVal.offset);
    }
    else {
      forest->RankProduce(tIdx, id, predIdx, lhId - id, splitVal.rankRange.rankLow, splitVal.rankRange.rankHigh);
    }
  }
}


/**
   @brief Absorbs the terminal list from a completed subtree.

   @param stTerm are subtree-relative indices.  These must be mapped to
   sample indices if the subtree is proper.

   @return void, with side-effected frontier map.
 */
void PreTree::SubtreeFrontier(const std::vector<TermKey> &stKey, const std::vector<unsigned int> &stTerm) {
  for (auto & key : stKey) {
    termKey.push_back(key);
  }

  for(auto & stIdx : stTerm) {
    termST.push_back(stIdx);
  }
}


/**
   @brief Constructs mapping from sample indices to leaf indices.

   @param tIdx is the index of the tree being produced.

   @return Reference to rewritten map, with side-effected Forest.
 */
const std::vector<unsigned int> PreTree::FrontierToLeaf(ForestTrain *forest, unsigned int tIdx) {
  std::vector<unsigned int> frontierMap(termST.size());
  unsigned int leafIdx = 0;
  unsigned int check = 0;
  for (auto & key : termKey) {
    for (unsigned int idx = key.base ; idx < key.base + key.extent; idx++) {
      unsigned int stIdx = termST[idx];
      frontierMap[stIdx] = leafIdx;
      check++;
    }
    forest->LeafProduce(tIdx, key.ptId, leafIdx++);
  }

  return frontierMap;
}


/**
   @return BV-aligned length of used portion of split vector.
 */
unsigned int PreTree::BitWidth() {
  return BV::SlotAlign(bitEnd);
}
