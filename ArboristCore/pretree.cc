// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "index.h"
#include "pretree.h"
#include "train.h"
#include "predictor.h"
#include "response.h"

#include <iostream>
using namespace std;

PreTree *PreTree::preTree = 0;
int PreTree::bitLength = -1;
int PreTree::ptCount = -1;
int PreTree::treeHeight = -1;
int PreTree::levelBase = -1;
int PreTree::leafCount = -1;
int PreTree::splitCount = -1;
int PreTree::treeBitOffset = 0;
bool *PreTree::treeSplitBits = 0;
int *PreTree::qOff = 0;
int *PreTree::qRanks = 0;
int *PreTree::sample2PT = 0;

// Quantiles can be derived by noting rank population (sCount) at each
// leaf.  After decision tree walked, leaf rank vectors are summed and
// the appropriate rank quantiles can be derived by walking the leaf sums
// in rank order.  Absolute quantiles can then be derived in a single
// pass using a rank2row mapping for the response.
//
// Leaf accumulators are not reused, so there is no need to record
// sample indices or ranks until the final row has been visited.
//

// Records terminal-node information for elements of the next level in the pre-tree.
// A terminal node may later be converted to non-terminal if found to be splitable.
// Initializing as terminal by default offers several advantages, such as avoiding
// the need to revise dangling non-terminals from an earlier level.
//
void PreTree::TreeInit(int _levelMax, int _bagCount) {
  if (ptCount < 0)
    ptCount = 2 * _levelMax; // Initial size estimate.

  preTree = new PreTree[ptCount];
  sample2PT = new int[_bagCount];
  for (int i = 0; i < _bagCount; i++) {
    sample2PT[i] = 0; // Unique root nodes zero.
  }
  preTree[0].lhId = -1;
  levelBase = 0;
  treeHeight = leafCount = 1;

  // Should be wide enough to hold all factor bits for an entire tree:
  //    #nodes * width of widest factor.
  //
  if (Predictor::NPredFac() > 0) {
    bitLength = 2 * _levelMax * Predictor::MaxFacCard();
    treeSplitBits = BitFactory(bitLength);
  }
}


void PreTree::TreeClear() {
  if (Predictor::NPredFac() > 0) {
    delete [] treeSplitBits;
    treeSplitBits = 0;
  }
  delete [] preTree;
  delete [] sample2PT;
  sample2PT = 0;
  preTree = 0;
  levelBase = bitLength = treeHeight = leafCount = -1;
}


// Assumes non-root.  Speculatively sets the two offspring slots as terminal lh, rh,
// respectively.
//
void PreTree::TerminalOffspring(int _parId, int &ptLH, int &ptRH) {
  ptLH = treeHeight++;
  preTree[_parId].lhId = ptLH;
  preTree[ptLH].lhId = -1;

  ptRH = treeHeight++;
  preTree[ptRH].lhId = -1;

  leafCount += 2;
}

// Fills in some fields for (generic) node found splitable.
//
void PreTree::NonTerminalGeneric(int _id, double _info, double _splitVal, int _predIdx) {
  PreTree *ptS = &preTree[_id];
  ptS->predIdx = _predIdx;
  ptS->info = _info;
  ptS->splitVal = _splitVal;
  leafCount--;
}

// Updates the high watermark for the preTree[] vector.  Forces a reallocation to
// twice the existing size, if necessary.
//
// N.B.:  reallocations incur considerable resynchronization costs if precipitated
// from the coprocessor.
//
void PreTree::CheckStorage(int splitNext, int leafNext) {
  if (treeHeight + splitNext + leafNext > ptCount)
    ReFactory();
  if (Predictor::NPredFac() > 0) {
    if (treeBitOffset + splitNext * Predictor::MaxFacCard() > bitLength)
      ReBits();
  }
}

// Guestimates safe height by doubling high watermark.
//
void PreTree::ReFactory() {
  ptCount <<= 1;
  PreTree *PTtemp = new PreTree[ptCount];
  for (int i = 0; i < treeHeight; i++)
    PTtemp[i] = preTree[i];

  delete [] preTree;
  preTree = PTtemp;
}

// Tree split bits accumulate, so data must be copied on realloc.
//
void PreTree::ReBits() {
  bitLength <<= 1;
  bool *TStemp = BitFactory(bitLength);
  for (int i = 0; i < treeBitOffset; i++)
    TStemp[i] = treeSplitBits[i];
  delete [] treeSplitBits;
  treeSplitBits = TStemp;
}

// Sets bit at current offset plus 'pos' to true.  Assumes that the bits have been
// initialized to false.
//
void PreTree::SingleBit(int pos) {
  //cout << "\t" << treeBitOffset << " + " << pos << endl;
  treeSplitBits[treeBitOffset + pos] = true;
}

void PreTree::NonTerminalFac(int treeId, double info, int predIdx) {
  double sval = treeBitOffset;
  treeBitOffset += Predictor::FacCard(predIdx);
  NonTerminalGeneric(treeId, info, sval, predIdx);
}


// Writes factor bits from all levels into contiguous vector and returns
// bit vector to uninitialized state.
//
// N.B.:  Should not be called unless FacWidth() > 0.
//
void PreTree::ConsumeSplitBits(int outBits[]) {
  for (int i = 0; i < treeBitOffset; i++) {
    outBits[i] = treeSplitBits[i]; // Upconverts to integer type for output to front end.
    //    cout << outBits[i] << endl;    
  }
  delete [] treeSplitBits;
  treeSplitBits = 0;
  treeBitOffset = 0;
}


// Consumes pretree nodes into the vectors needed by the decision tree.
//
// Assigns a breadth-first numbering to minimize branching deltas.
//
// Returns the tree size, which is the maximum offset filled in.
//
void PreTree::ConsumeNodes(int leafPred, int predVec[], double splitVec[], int bumpVec[], double scoreVec[]) {
  Response::ProduceScores(treeHeight, scoreVec);
  for (int idx = 0; idx < treeHeight; idx++) {
      if (IsNT(idx)) { // Consumes splits.
	PreTree ptNode = preTree[idx];
	predVec[idx] = ptNode.predIdx;
	splitVec[idx] = ptNode.splitVal;
	bumpVec[idx] = ptNode.lhId - idx;
      }
      else { // Consumes leaves.
	predVec[idx] = leafPred;
      }
  }
  //  cout << leafCount << " leaves on tree height " << treeHeight << endl;
}

// Same as Sample2PT(), but asserts terminality.
//
int PreTree::Sample2Leaf(int sIdx) {
  int leafIdx = Sample2PT(sIdx);
  if (IsNT(leafIdx)) // ASSERTION
    cout << "Unexpected non-terminal:  " << leafIdx << endl;

  return leafIdx;
}
