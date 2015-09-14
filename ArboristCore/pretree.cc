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

#include "pretree.h"
#include "predictor.h"
#include "samplepred.h"

//#include <iostream>
using namespace std;


// Records terminal-node information for elements of the next level in the pre-tree.
// A terminal node may later be converted to non-terminal if found to be splitable.
// Initializing as terminal by default offers several advantages, such as avoiding
// the need to revise dangling non-terminals from an earlier level.
//

int PreTree::nPred = 0;
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
PreTree::PreTree(int _bagCount) {
  bagCount = _bagCount;
  sample2PT = new int[bagCount];
  for (int i = 0; i < bagCount; i++) {
    sample2PT[i] = 0;
  }
  nodeCount = heightEst;   // Initial height estimate.
  nodeVec = new PTNode[nodeCount];
  nodeVec[0].id = 0;
  nodeVec[0].lhId = -1;
  info = new double[nPred];
  for (int i = 0; i < nPred; i++)
    info[i] = 0.0;
  treeHeight = leafCount = 1;
  treeBitOffset = 0;
  treeSplitBits = BitFactory();
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
   @brief Allocates the bit string for the current (pre)tree and initializes to false.

    // Should be wide enough to hold all factor bits for an entire tree:
    //    #nodes * width of widest factor.
    //

   @return pointer to allocated vector.
*/
bool *PreTree::BitFactory(int _bitLength) {
  bool *tsb = 0;
  if (Predictor::NPredFac() > 0) {
    bitLength = _bitLength == 0 ? nodeCount * Predictor::MaxFacCard() : _bitLength;
    tsb = new bool[bitLength];
    for (int i = 0; i < bitLength; i++)
      tsb[i] = false;
  }

  return tsb;
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
   @brief Speculatively sets the two offspring slots as terminal lh, rh.

   @param _parId is the pretree index of the parent.

   @param ptLH outputs the left-hand node index.

   @param ptRH outputs the right-hand node index.

   @return void, with output reference parameters.
*/
void PreTree::TerminalOffspring(int _parId, int &ptLH, int &ptRH) {
  ptLH = treeHeight++;
  nodeVec[_parId].lhId = ptLH;
  nodeVec[ptLH].id = ptLH;
  nodeVec[ptLH].lhId = -1;

  ptRH = treeHeight++;
  nodeVec[ptRH].id = ptRH;
  nodeVec[ptRH].lhId = -1;

  leafCount += 2;
}

/**
   @brief Fills in some fields for (generic) node found splitable.

   @param _id is the node index.

   @param _info is the information content.

   @param _splitVal is the splitting value.

   @param _predIdx is the splitting predictor index.

   @return void.
*/
void PreTree::NonTerminal(int _id, double _info, double _splitVal, int _predIdx) {
  PTNode *ptS = &nodeVec[_id];
  ptS->predIdx = _predIdx;
  ptS->splitVal = _splitVal;
  info[_predIdx] += _info;
  leafCount--;
}


double PreTree::Replay(SamplePred *samplePred, int predIdx, int level, int start, int end, int ptId) {
  return samplePred->Replay(sample2PT, predIdx, level, start, end, ptId);
}


/**
   @brief Updates the high watermark for the preTree vector.  Forces a reallocation to
   twice the existing size, if necessary.

   N.B.:  reallocations incur considerable resynchronization costs if precipitated
   from the coprocessor.

   @param levelWidth is the count of nodes in the next level.

   @return void.
*/
void PreTree::CheckStorage(int splitNext, int leafNext) {
  if (treeHeight + splitNext + leafNext > nodeCount)
    ReNodes();
  if (Predictor::NPredFac() > 0) {
    if (treeBitOffset + splitNext * Predictor::MaxFacCard() > bitLength)
      ReBits();
  }
}


/**
 @brief Guestimates safe height by doubling high watermark.

 @return void.
*/
void PreTree::ReNodes() {
  nodeCount <<= 1;
  PTNode *PTtemp = new PTNode[nodeCount];
  for (int i = 0; i < treeHeight; i++)
    PTtemp[i] = nodeVec[i];

  delete [] nodeVec;
  nodeVec = PTtemp;
}


/**
 @brief Tree split bits accumulate, so data must be copied on realloc.

 @return void.
*/
void PreTree::ReBits() {
  bool *TStemp = BitFactory(bitLength << 1);
  for (int i = 0; i < treeBitOffset; i++)
    TStemp[i] = treeSplitBits[i];
  delete [] treeSplitBits;
  treeSplitBits = TStemp;
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
void PreTree::DecTree(int predTree[], double splitTree[], int bumpTree[], unsigned int *facBits, double predInfo[]) {
  SplitConsume(predTree, splitTree, bumpTree);
  BitConsume(facBits);
  for (int i = 0; i < nPred; i++)
    predInfo[i] += info[i];
}


/**
   @brief Consumes nonterminal information into the dual-use vectors needed by the decision tree.  Leaf information is post-assigned by the response-dependent Sample methods.

   @param nodeVal outputs splitting predictor / leaf extent : nonterminal / terminal.

   @param numVec outputs splitting value / leaf score : nonterminal / terminal.

   @param bumpVec outputs the left-hand node increment:  nonzero iff nonterminal.

   @return void, with output reference parameter vectors.
*/
void PreTree::SplitConsume(int nodeVal[], double numVec[], int bumpVec[]) {
  for (int idx = 0; idx < treeHeight; idx++) {
    nodeVec[idx].SplitConsume(nodeVal[idx], numVec[idx], bumpVec[idx]);
  }
}


/**
  @brief Writes factor bits from all levels into contiguous vector and resets bit state.

  @param outBits outputs the local bit values.
  
  @return void, with output parameter vector.
*/
void PreTree::BitConsume(unsigned int *outBits) {
  if (treeBitOffset != 0) {
    for (int i = 0; i < treeBitOffset; i++) {
      outBits[i] = treeSplitBits[i]; // Upconverts to output type.
    }
    delete [] treeSplitBits;
  }
}


/**
   @brief Consumes the node fields of nonterminals (splits).

   @param pred outputs the splitting predictor, if a split.

   @param num outputs the splitting value, if a split.

   @param bump outputs the distance to the left-hand subnode, if a split.

   @return void.
 */
void PTNode::SplitConsume(int &pred, double &num, int &bump) {
  if (lhId > id) {
    pred = predIdx;
    num = splitVal;
    bump = lhId - id;
  }
}
