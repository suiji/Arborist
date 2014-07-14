/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "level.h"
#include "pretree.h"
#include "dataord.h"
#include "dectree.h"
#include "train.h"
#include "splitsig.h"
#include "response.h"

#include <iostream>
using namespace std;

int PreTree::bagCount = -1;
int *PreTree::qOff = 0;
int *PreTree::qRanks = 0;
DevSplit *PreTree::devSplit = 0;

// Quantiles can be derived by noting rank population (sCount) at each
// leaf.  After decision tree walked, leaf rank vectors are summed and
// the appropriate rank quantiles can be derived by walking the leaf sums
// in rank order.  Absolute quantiles can then be derived in a single
// pass using a rank2row mapping for the response.
//
// Leaf accumulators are not reused, so there is no need to record
// sample indices or ranks until the final row has been visited.
//
int PreTree::rowBlock = -1;
SplitNode **PreTree::splitSet = 0;
Leaf **PreTree::leafSet = 0;
int PreTree::leafCount = -1;
int PreTree::splitCount = -1;
SplitNode **PreTree::parent = 0;
SplitNode **PreTree::parentNext = 0;
int *PreTree::leafMap = 0;

void PreTree::Factory(int nSamp, int _accumCount, int _rowBlock) {
  rowBlock = _rowBlock;
  leafSet = new Leaf*[nSamp];
  devSplit = new DevSplit[_rowBlock]; // Built on device.
  splitSet = new SplitNode*[nSamp];
  parent = new SplitNode*[_accumCount]; // TODO:  May be unnecessary.
  parentNext = new SplitNode*[_accumCount]; // TODO:  " " 
  leafMap = new int[_rowBlock];
}

void PreTree::DeFactory() {
  delete [] devSplit;
  delete [] splitSet;
  delete [] leafSet;
  delete [] parent;
  delete [] parentNext;
  delete [] leafMap;
  devSplit = 0;
  splitSet = 0;
  leafSet = 0;
  parent = 0;
  parentNext = 0;
  leafCount = -1;
  splitCount = -1;
  leafMap = 0;
}

void PreTree::TreeInit(const int _bagCount, const int _accumCount) {
  bagCount = _bagCount;
  leafCount = 0;
  splitCount = 0;
  for (int i = 0; i < bagCount; i++) // Necessary?
    splitSet[i] = 0;
  parentNext[0] = 0;
  parent[0] = 0;

  SplitSigFac::TreeInit(); // Tree bit offsets managed entirely by pretree.
}

// Only the root has empty parent node.
//
PTNode::PTNode(class SplitNode *_par, const bool isLH) : par(_par) , treeOff(-1) {
  if (_par != 0) {
    depth = 1 + _par->depth;
    if (isLH)
      _par->lh = this;
    else
      _par->rh = this;
  }
  else
    depth = 0;
}

// Split nodes are created as result of lowering the SplitSig associated a
// given accumulator ('liveIdx') and predictor ('predIdx').
SplitNode *PreTree::AddSplit(const int liveIdx, int predIdx, char subset, double gini, bool isLH) {
  return AddSplit(parent[liveIdx], predIdx, subset, gini, isLH);
}

SplitNode *PreTree::AddSplit(SplitNode *par, int predIdx, char subset, double gini, bool isLH) {
  SplitVal sval;
  sval.num = SplitSigFac::TreeBitOffset();
  SplitNode *splitNode = new SplitNode(-(1+predIdx), sval, gini, par, isLH);
  splitSet[splitCount++] = splitNode;

  //  cout << "Adding split at index " << splitCount << ", depth:  " << splitNode->depth << endl;

  // Cannot produce new 'treeBitOffset' value until after current value has been consumed (above);
  SplitSigFac::SplitBits(subset);

  splitNode->id = splitCount-1; // DIAGNOSTIC
  return splitNode;
}

// For now, 'parent[]' maps accumulator (node) indices to their counterparts in the pre-tree.
// Avoids the need to maintain a live pointer into the pretree, but suffers from problems of
// its own.
//
void PreTree::SetParent(const int lhId, const int rhId, SplitNode *splitNode) {
  //  cout << "Split for upcoming " << lhId << " / " << rhId << endl;
  if (lhId >= 0)
    parentNext[lhId] = splitNode;
  if (rhId >= 0)
    parentNext[rhId] = splitNode;
}

void PreTree::FlushLevel(const int countNext) {
  for (int i = 0; i < countNext; i++) {
    parent[i] = parentNext[i];
    parentNext[i] = 0; // Necessary?
  }
}

double PreTree::ParGini(const int liveIdx) {
  return parent[liveIdx] == 0 ? 0.0 : parent[liveIdx]->Gini;
}

// Leaf entry point for non-splitting nodes.  Finalized leaf construction:  all
// relevant fields available from NodeCache.
//
// Handler determines which fields to set.
//
int PreTree::AddLeaf(const int liveIdx, NodeCache *tfAccum) {
  int id = AddLeaf(parent[liveIdx], tfAccum->isLH);

  return id;
}

// Adds an unfinalized leaf, i.e., a leaf for which the split-determined offsets are
// not yet known.  These leaves return an encoded offset enabling the caller to
// distinguish pre-tree leaves from split nodes.
//
int PreTree::AddLeaf(SplitNode *par, bool _isLH) {
  int leafOff = leafCount++;
  // ASSERTION:
  //  if (leafCount > bagCount)
  //cout << "More leaves than nodes" << endl;

  Leaf *leaf = new Leaf(par, _isLH);
  leafSet[leafOff] = leaf;
  int id = -(leafOff + 1);

  // Leaf offsets are coded as negative values in order to distinguish from predictor
  // (splitting) indices.  This operation is idempotent, so is applied identically for
  // encoding and decoding.
  //
  leaf->id = id;// DIAGNOSTIC
  return id;
}

// Variant in which 'leafId' already determined.
// Only client is coprocessor version, in which leaf offsets assigned on device.
//
// Leaf offsets are coded as negative values in order to distinguish from predictor
// (splitting) indices.  This operation is idempotent, so is applied identically// for encoding and decoding.
//
void PreTree::AddLeaf(SplitNode *par, int leafId, bool _isLH) {
  int leafOff = -(leafId + 1);
  leafCount++;

  // ASSERTION:
  //  if (leafCount > bagCount)
  //cout << "More leaves than nodes" << endl;

  leafSet[leafOff] = new Leaf(par, _isLH);
}


// Flushes out the 'dSplitCount' records read from the device.
//
int PreTree::Produce(int levels) {
  int dSplitCount;
  LevelCoproc::ReadPreTree(&dSplitCount, devSplit);
  splitCount = leafCount = 0;//TEMPORARY
  for (int i = 0; i < dSplitCount; i++) {
    DevSplit *dSplit = &devSplit[i];
    int parId = dSplit->parId;

    SplitNode *par = parId < 0 ? 0 : splitSet[parId];
    // if (parId >= i) // ASSERTION
   //cout << "Invalid parent" << endl;
    SplitNode *preTree = AddSplit(par, dSplit->pred, dSplit->subset, dSplit->Gini, dSplit->isLH);
    int lhOff = dSplit->lhOff;
    if (lhOff < 0) {
      AddLeaf(preTree, lhOff, true);
    }
    //    else if (lhOff > 255)
      //cout << "Uninitialized lhs" << endl;
    int rhOff = dSplit->rhOff;
    if (rhOff < 0) {
      AddLeaf(preTree, rhOff, false);
    }
    //    else if (rhOff > 255) // ASSERTION
    //cout << "Uninitialized rhs" << endl;
  }
  LevelCoproc::ReadLeafMap(leafMap, rowBlock);
  ResponseCtg::ScoreLeaves(leafMap, leafSet, leafCount, bagCount);

  return TreeOffsets(levels); // Moved here from DecTree::ConsumPretree()
}

// Dump:  check that parent's rh or lh references this id.
    //    cout << "host: " << splitSet[i]->depth << ", " << splitSet[i]->lh->id << " / " << splitSet[i]->rh->id << endl;
    //    cout << "Split " << i << ": " << dSplit->depth << ", " << dSplit->pred << ", " << parId << ", " << dSplit->lhOff << ", " << dSplit->rhOff << endl;

  /*
  cout << leafCount << " leaves: " << endl;
  for (int i = 0; i < leafCount; i++)
    cout << leafSet[i]->depth << endl;
  cout << "Splits:  " << endl;
  for (int i = 0; i < splitCount; i++) {
    SplitNode *split = splitSet[i];
    cout << split->depth << ", " << split->lh->depth << ", " << split->rh->depth << endl;
  }
  */


// Returns the tree size, which is the maximum offset filled in.
//
int PreTree::TreeOffsets(const int levels) {
  if (splitCount == 0) { // Nothing else to do.
    leafSet[0]->treeOff = 0;
    return 1;
  }

  int leafIdx = 0;
  int splitIdx = 0;
  int offset = 0;

  // Assigns a breadth-first numbering to minimize deltas.
  //
  //  cout << leafCount << " leaves, " << splitCount << " splits, " << levels << " levels" <<endl;
  for (int l = 0; l < levels; l++) {
    //cout << "Level " << l << " splits: " << endl;
    // TODO:  Can higher-offset split indices have depth less than 'l'?
    while (splitIdx < splitCount && splitSet[splitIdx]->depth <= l) {
      SplitNode *split = splitSet[splitIdx];
      split->treeOff = offset++;
      /*
	SplitNode *par = split->par;
	cout << splitIdx << ": "<< offset-1 << " split par:  " << (par == 0 ? -1 : par->treeOff) <<endl;
      if (par != 0 && par->lh != split && par->rh != split)
	cout << "Unlinked parent"  << endl;
      if (split->lh->depth != l + 1 || split->rh->depth != l + 1)
	cout << "Unlinked child" << endl;
      */
      splitIdx++;
    }
    //   cout << "Level " << l << " leaves: " << endl;
    // Higher-offset leaf indices may have depth one less than 'l', as
    // nodes failing to split for Gini reasons are not identified until
    // the next level.
    while (leafIdx < leafCount && leafSet[leafIdx]->depth <= l) {
      Leaf *leaf = leafSet[leafIdx];
      leaf->treeOff = offset++;
      /*
      SplitNode *par = leaf->par;
	cout << leafIdx << ": " << offset-1 << " leaf par:  " << par->treeOff << endl;
      if (par->lh != leaf && par->rh != leaf)
	cout << "Unlinked leaf" << endl;
      */
      leafIdx++;
    }
  }
  //  cout << splitIdx << " splits and " << leafIdx << " leaves assigned offset" << endl;

  for (int i = 0; i < splitCount; i++) {
    //cout << "Split " << i  << ": " << splitSet[i]->lh << " , " << splitSet[i]->rh << endl;
    SplitNode *split = splitSet[i];
    int off = split->treeOff;
    if (off < 0) // ASSERTION
      cout << "Unexpected tree offset:  " << off << endl;
    split->bump.left = split->lh->treeOff - off;
    split->bump.right = split->rh->treeOff - off;
    /*
    if (split->lh->treeOff <= split->treeOff || split->rh->treeOff <= split->treeOff)
      cout << "Bad bump, level " << split->depth << ", " << split->treeOff << ":  " << split->lh->treeOff << " / " << split->rh->treeOff << endl;
    */
  }

  return offset;  // Height on tree replay:  one beyond highest used offset.
}

#ifdef DIAGNOSTIC
  // DIAGNOSTIC:
  //
  int *testCount = new int[offset];
  for (int i = 0; i <= offset; i++)
    testCount[i] = 0;
  for (int i = 0; i < splitCount; i++) {
    int off = splitSet[i]->treeOff;
    testCount[off]++;
  }
  for (int i = 0; i < leafCount; i++) {
    int off = leafSet[i]->treeOff;
    testCount[off++];
  }
  for (int i = 0; i <= offset; i++) {
    if (testCount[i] > 1)
      cout << "DUPLICATE OFFSET:  " << i << endl;
  }
#endif

void PreTree::ConsumeLeaves(double scoreVec[], int predVec[]) {
  for (int i = 0; i < leafCount; i++) {
    Leaf *leaf = leafSet[i];
    int treeOff = leaf->treeOff;
    scoreVec[treeOff] = leaf->score;
    predVec[treeOff] = DecTree::leafPred;
    delete leaf;
  }
  leafCount = 0;
}

void PreTree::ConsumeSplits(double splitVec[], int predVec[], Bump bumpVec[]) {
  for (int i = 0; i < splitCount; i++) {
    SplitNode *ss = splitSet[i];
    int off = ss->treeOff;
    predVec[off] = ss->pred;
    splitVec[off] = ss->sval.num;
    bumpVec[off] = ss->bump;
    delete ss;

    // ASSERTION:
    //    if (predVec[off] == leafPred)
    //cout << "Split with no predictor" << endl;
  }
  splitCount = 0;
}


// Copies leaf information into leafOff[] and ranks[].
//
void PreTree::DispatchQuantiles(const int treeSize, int leafPos[], int leafExtent[], int rank[], int rankCount[]) {
  // Must be wide enough to access all decision-tree offsets.
  int *seen = new int[treeSize];
  for (int i = 0; i < treeSize; i++) {
    seen[i] = 0;
    leafExtent[i] = -1;
    leafPos[i] = -1;
  }

  int totCt = 0;
  for (int i = 0; i < leafCount; i++) {
    int treeOff = leafSet[i]->treeOff;
    int ct = leafSet[i]->extent;
    leafExtent[treeOff] = ct;
    leafPos[treeOff] = totCt;
    totCt += ct;
  }

  // ASSERTION
  if (totCt != bagCount)
    cout << "Leaf count " << totCt << " != " << "bag count " << bagCount << endl;

  for (int i = 0; i < bagCount; i++) {
    int rk = AccumHandlerReg::sample2Rank[i];
    // ASSERTION:
    //    if (rk > Predictor::nRow)
    //  cout << "Invalid rank:  " << rk << " / " << Predictor::nRow << endl;
    int leafId = leafSet[AccumHandler::sample2Accum[i]]->treeOff;
    int rkOff = leafPos[leafId] + seen[leafId];
    rank[rkOff] = rk;
    rankCount[rkOff] = AccumHandlerReg::sample[i].rowRun;
    seen[leafId]++;
  }

  delete [] seen;
}
