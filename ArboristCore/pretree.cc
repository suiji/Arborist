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
#include "samplepred.h"

//#include <iostream>
using namespace std;


// Records terminal-node information for elements of the next level in the pre-tree.
// A terminal node may later be converted to non-terminal if found to be splitable.
// Initializing as terminal by default offers several advantages, such as avoiding
// the need to revise dangling non-terminals from an earlier level.
//

unsigned int PreTree::nPred = 0;
unsigned int PreTree::heightEst = 0;

/**
   @brief Caches the row count and computes an initial estimate of node count.

   @param _nSamp is the number of samples.

   @param _minH is the minimal splitable index node size.

   @return void.
 */
void PreTree::Immutables(unsigned int _nPred, unsigned int _nSamp, unsigned int _minH) {
  nPred = _nPred;

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
  nPred = heightEst = 0;
}


/**
   @brief Per-tree initializations.

   @param _treeBlock is the number of PreTree initialize.

   @return void.
 */
PreTree::PreTree(int _bagCount) : height(1), leafCount(1), bitEnd(0), bagCount(_bagCount) {
  sample2PT = new unsigned int[bagCount];
  for (int i = 0; i < bagCount; i++) {
    sample2PT[i] = 0;
  }
  nodeCount = heightEst;   // Initial height estimate.
  nodeVec = new PTNode[nodeCount];
  nodeVec[0].id = 0; // Root.
  nodeVec[0].lhId = 0; // Initializes as terminal.
  info = new double[nPred];
  for (unsigned int i = 0; i < nPred; i++)
    info[i] = 0.0;
  splitBits = BitFactory();
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
void PreTree::RefineHeight(unsigned int height) {
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
  return new BV(nodeCount * PBTrain::CardMax());
}


/**
   @brief Per-tree finalizer.
 */
PreTree::~PreTree() {
  delete [] nodeVec;
  delete [] sample2PT;
  delete [] info;
}

/**
   @brief Speculatively sets the two offspring slots as terminal lh, rh and changes status of this from terminal to nonterminal.

   @param _parId is the pretree index of the parent.

   @param ptLH outputs the left-hand node index.

   @param ptRH outputs the right-hand node index.

   @return void, with output reference parameters.
*/
void PreTree::TerminalOffspring(unsigned int _parId, unsigned int &ptLH, unsigned int &ptRH) {
  ptLH = height++;
  nodeVec[_parId].lhId = ptLH;
  nodeVec[ptLH].id = ptLH;
  nodeVec[ptLH].lhId = 0;

  ptRH = height++;
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
void PreTree::NonTerminalFac(double _info, unsigned int _predIdx, unsigned int _id, unsigned int &ptLH, unsigned int &ptRH) {
  TerminalOffspring(_id, ptLH, ptRH);
  PTNode *ptS = &nodeVec[_id];
  ptS->predIdx = _predIdx;
  info[_predIdx] += _info;
  ptS->splitVal.offset = bitEnd;
  bitEnd += PBTrain::FacCard(_predIdx);
}


/**
   @brief Finalizes numeric-valued nonterminal.

   @param _info is the splitting information content.

   @param _predIdx is the splitting predictor index.

   @param _id is the node index.

   @return void.
*/
void PreTree::NonTerminalNum(double _info, unsigned int _predIdx, unsigned int _rkLow, unsigned int _rkHigh, unsigned int _id, unsigned int &ptLH, unsigned int &ptRH) {
  TerminalOffspring(_id, ptLH, ptRH);
  PTNode *ptS = &nodeVec[_id];
  ptS->predIdx = _predIdx;
  ptS->splitVal.rkMean = 0.5 * (double(_rkLow) + double(_rkHigh));
  info[_predIdx] += _info;
}


double PreTree::Replay(SamplePred *samplePred, unsigned int predIdx, int level, int start, int end, unsigned int ptId) {
  return samplePred->Replay(sample2PT, predIdx, level, start, end, ptId);
}


/**
   @brief Updates the high watermark for the preTree vector.  Forces a
   reallocation to twice the existing size, if necessary.

   N.B.:  reallocations incur considerable resynchronization costs if
   precipitated from the coprocessor.

   @param splitNext is the count of splits in the upcoming level.

   @param leafNext is the count of leaves in the upcoming level.

   @return void.
*/
void PreTree::CheckStorage(int splitNext, int leafNext) {
  if (height + splitNext + leafNext > nodeCount)
    ReNodes();

  unsigned int bitMin = bitEnd + splitNext * PBTrain::CardMax();
  if (bitMin > 0) {
    splitBits = splitBits->Resize(bitMin);
  }
}


/**
 @brief Guestimates safe height by doubling high watermark.

 @return void.
*/
void PreTree::ReNodes() {
  nodeCount <<= 1;
  PTNode *PTtemp = new PTNode[nodeCount];
  for (int i = 0; i < height; i++)
    PTtemp[i] = nodeVec[i];

  delete [] nodeVec;
  nodeVec = PTtemp;
}


/**
  @brief Consumes all pretree nonterminal information into crescent decision forest.

  @param predTree outputs leaf width for terminals, predictor index for nonterminals.

  @param splitTree outputs score for terminals, splitting value for nonterminals.

  @param bumpTree outputs zero for terminals, index delta for nonterminals.

  @param facBits outputs the splitting bits.

  @param predInfo accumulates the information contribution of each predictor.

  @return void, with output vector parameters.
*/
void PreTree::DecTree(std::vector<unsigned int> &predTree, std::vector<double> &splitTree, std::vector<unsigned int> &bumpTree, std::vector<unsigned int> &facBits, double predInfo[]) {
  NodeConsume(predTree, splitTree, bumpTree);
  splitBits->Consume(facBits, bitEnd);
  delete splitBits;

  for (unsigned int i = 0; i < nPred; i++)
    predInfo[i] += info[i];
}


/**
   @brief Consumes nonterminal information into the dual-use vectors needed by the decision tree.  Leaf information is post-assigned by the response-dependent Sample methods.

   @param nodeVal outputs splitting predictor / leaf extent : nonterminal / terminal.

   @param numVec outputs splitting value / leaf score : nonterminal / terminal.

   @param bumpVec outputs the left-hand node increment:  nonzero iff nonterminal.

   @return void, with output reference parameter vectors.
*/
void PreTree::NodeConsume(std::vector<unsigned int> &pred, std::vector<double> &split, std::vector<unsigned int> &bump) {
  for (int idx = 0; idx < height; idx++) {
    nodeVec[idx].Consume(pred, split, bump);
  }
}


/**
   @return aligned length of used portion of split vector.
 */
unsigned int PreTree::BitWidth() {
  return BV::SlotAlign(bitEnd);
}


/**
   @brief Consumes the node fields of nonterminals (splits).

   @param pred outputs the splitting predictor, if a split.

   @param num outputs the splitting value, if a split.

   @param bump outputs the distance to the left-hand subnode, if a split.

   @return void.
 */
void PTNode::Consume(std::vector<unsigned int> &pred, std::vector<double> &num, std::vector<unsigned int> &bump) {
  bool nonTerminal = lhId > 0;
  bump.insert(bump.end(), 1, nonTerminal ? lhId - id : 0);
  pred.insert(pred.end(), 1, nonTerminal ? predIdx : 0);
  num.insert(num.end(), 1, nonTerminal ? (PredBlock::IsFactor(predIdx) ? splitVal.offset : splitVal.rkMean) : 0.0);
}
