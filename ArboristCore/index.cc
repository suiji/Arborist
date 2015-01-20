// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "index.h"
#include "splitsig.h"
#include "pretree.h"
#include "response.h"
#include "train.h"
#include "splitpred.h"
#include "samplepred.h"

#include <iostream>
using namespace std;


IndexNode *IndexNode::indexNode = 0;
int IndexNode::totLevels = -1;
int IndexNode::levelMax = -1;

int NodeCache::minHeight = -1;
NodeCache *NodeCache::nodeCache = 0;

//
void IndexNode::Factory(int _minHeight, int _totLevels) {
  totLevels = _totLevels;

  NodeCache::Factory(_minHeight);
}

void IndexNode::TreeInit(int _levelMax, int bagCount, int nSamp, double sum) {
  levelMax = _levelMax;
  indexNode = new IndexNode[levelMax];
  NodeCache::TreeInit();
  SplitPred::TreeInit();
  SplitSig::TreeInit(levelMax);
  NextLevel(0, 0, bagCount, nSamp, sum, 0.0);
}

void NodeCache::TreeInit() {
  nodeCache = new NodeCache[levelMax];
}

void NodeCache::TreeClear() {
  delete [] nodeCache;
  nodeCache = 0;
}

void IndexNode::TreeClear() {
  NodeCache::TreeClear();
  delete [] indexNode;
  indexNode = 0;
}

// Updates 'levelMax' and data structures depending upon it.
// Invokes ReFactory() methods on only those classes which must be integral
// prior to NodeCache's consumption.
//
void IndexNode::ReFactory() {
  levelMax = Train::ReFactory();

  delete [] indexNode;
  indexNode = new IndexNode[levelMax];

  // RestageMap:  restageMap[]
  RestageMap::ReFactory(levelMax);
}


// Reallocation of NodeCache, as well as classes not required for consumption of
// all instances.
// 
// By delaying NodeCache reallocation until after consumption, content need not
// be copied.  This includes SplitSig references recorded directly on the cached
// node.
//
void NodeCache::ReFactory() {
  delete [] nodeCache;
  nodeCache = new NodeCache[levelMax];

  // Response drives SPCtg and SPReg reallocations.
  //
  Response::ReFactory(levelMax);

  // SplitPred:  splitFlags[], runFlags[]
  //
  SplitPred::ReFactory(levelMax);

  //  levelSS[]
  //
  SplitSig::ReFactory(levelMax);
}

void IndexNode::DeFactory() {
  delete [] indexNode;
  indexNode = 0;
  levelMax = -1;
  totLevels = -1;
  NodeCache::DeFactory();
}

void NodeCache::Factory(int _minHeight) {
  minHeight = _minHeight;
}

void NodeCache::DeFactory() {
  minHeight = -1;
}

// Monolith entry point for per-level splitting.
//
// Returns count of levels.
//
int IndexNode::Levels() {
  int splitCount = 1;// Single root node at level zero.
  for (int level = 0; splitCount > 0 && (totLevels == 0 || level < totLevels); level++) {
    Response::LevelSums(splitCount);
    LateFields(splitCount);
    SplitPred::Level(splitCount, level);
    int lhSplitNext, rhSplitNext, leafNext;
    int totLhIdx = NodeCache::InterLevel(level, splitCount, lhSplitNext, rhSplitNext, leafNext);
    int splitNext = lhSplitNext + rhSplitNext;
    bool reFac = CheckStorage(splitCount, splitNext, leafNext);
    NodeCache::NextLevel(splitCount, lhSplitNext, totLhIdx, reFac);
    splitCount = splitNext;
  }

  return PreTree::TreeHeight();
}

// Caches all node information from the current level into NodeCache workspace.
// This circumvents crosstalk as the next level's nodes are populated.
//
void NodeCache::CacheNodes(int splitCount) {
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    Cache(splitIdx);
  }
}

// Performs the inter-level arbitration needed to initilialize split nodes for
// the next level.
//
// Returns start of RH position for restage partioning, which takes place
// just before the next level is split.
//
// TODO:  MUST guarantee that no zero-length "splits" have been introduced.
// Not only are these nonsensical, but they are also dangerous, as they violate
//  various assumptions about the integrity of the intermediate respresentation.
//
int NodeCache::InterLevel(int level, int splitCount, int &lhSplitNext, int &rhSplitNext, int &leafNext) {
  // The arg-max calls operate on distinct vectors.  Assuming that
  // # samples >> # predictors, parallelization of this loop
  // is warranted as 'splitCount' grows.
  //
  CacheNodes(splitCount);
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++)
    nodeCache[splitIdx].Splitable(level);

  // Restaging is implemented as a stable partition, and is faciliated by
  // enumerating all left-hand subnodes before the first right-hand subnode.
  //
  lhSplitNext = 0;
  rhSplitNext = 0;
  leafNext = 0;
  int rhStart = 0;
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++)
    rhStart += nodeCache[splitIdx].SplitCensus(lhSplitNext, rhSplitNext, leafNext);

  return rhStart;
}

// Ensures that sufficient space is present in the index tree and PreTree to
// accomodate the next level's nodes.
//
// Returns true iff IndexNode reallocation takes place.
//
bool IndexNode::CheckStorage(int splitCount, int splitNext, int leafNext) {
  bool reFac;
  if (splitNext > levelMax) {
    reFac = true;
    ReFactory();//splitCount);
  }
  else
    reFac = false;

  PreTree::CheckStorage(splitNext, leafNext);
  return reFac;
}

// Walks the list of cached splits from the level just concluded, adding PreTree
// terminals and IndexNodes for the next level.
//
// Informs PreTree to revise level-based information.
//
void NodeCache::NextLevel(int splitCount, int lhSplitNext, int totLhIdx, bool reFac) {
  PreTree::NextLevel();
  RestageMap::Commence(splitCount, totLhIdx);

  int lhCount = 0;
  int rhCount = 0;
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++)
    nodeCache[splitIdx].Consume(lhSplitNext, lhCount, rhCount);

  if (lhSplitNext != lhCount) // ASSERTION
    cout << "Next level split-count mismatch" << endl;

  // Reallocates vectors potentially referenced by objects in the node cache.
  //
  if (reFac)
    ReFactory();
}

// Checks for information content and, if found, updates the serialized
// pre-tree with splitting information.
//
void NodeCache::Splitable(int level) {
  int splitIdx = this - nodeCache;
  splitSig = SplitSig::ArgMax(splitIdx, level, preBias, minInfo);
  if (splitSig != 0) // Tags PreTree node as splitable.
    splitSig->NonTerminal(splitIdx, level,  ptId, lhStart);
  else
    ptL = ptR = -1;  // Prevents restaging.
}


// If the cached node splits, then a census is taken of the next level's
// left and right split nodes nodes and its leaves.
//
// Returns count of indices subsumed by LH, so that caller can compute the
// total LH extent for the next level.
//
int NodeCache::SplitCensus(int &lhSplitNext, int &rhSplitNext, int &leafNext) {
  int lhIdxNext = 0;

  if (splitSig != 0) {
    int lhSCount, lhIdxCount;
    splitSig->LHSizes(lhSCount, lhIdxCount);

    if (TerminalSize(lhSCount, lhIdxCount)) {
      leafNext++;
    }
    else {
      lhIdxNext = lhIdxCount;
      lhSplitNext++;
    }

    if (TerminalSize(sCount - lhSCount, idxCount - lhIdxCount)) {
      leafNext++;
    }
    else
      rhSplitNext++;
  }

  return lhIdxNext;
}

// LH and RH pre-tree nodes are made for all split nodes actually found to be
// splitable during this interlevel pass.  Node indices for both sides are passed
// to the Replay().  Terminality constraints are checked and index tree nodes
// (IndexNode) are made for all sides not so constrained.  Split node order is
// assigned so as to correspond with expectations of restaging, which takes place
// at the start of the next level.
//
// The node should be considered dead at return, as all useful information will have
// been extracted for use elsewhere.
//
void NodeCache::Consume(int lhSplitNext, int &lhSplitCount, int &rhSplitCount) {
  int lhIdxCount = 0;
  int lNext = -1;
  int rNext = -1;
  if (splitSig != 0) {
    // Pre-tree nodes (PreTree) for both sides are initialized as leaves (i.e.,
    // terminal), and are only later updated as splits if found to be splitable
    // during the next inter-level pass.
    //
    PreTree::TerminalOffspring(ptId, ptL, ptR);

    // With LH and RH PreTree indices known, the sample indices associated with
    // this split node can be looked up and remapped.  Replay() assigns actual
    // index values, irrespective of whether the pre-tree nodes at these indices
    // are terminal or non-terminal.  Hence the current 'restage' values suffice
    // to convey this information, although these may later be reset to a negative
    // value if found to reference terminals.
    //
    double lhSum = splitSig->Replay(this - nodeCache, ptL, ptR);

    // Index tree nodes (IndexNode), OTOH, are only made for those sides with the
    // potential to split - that is, which are not already known to be terminal.
    // These are made by NextLevel(), which is invoked in a manner guaranteeing
    // an ordering in which left-hand splits precede right-hand splits.
    // This ordering ensures that offset values assigned to splits reflect the same
    // ordering as is assigned by restaging, which effects a stable partition of
    // this level's predictor sample orderings (SampleOrd).
    //
    // Sides flagged as terminal have the respective 'restage' field reset to a
    // negative value.  This ensures that restaging does not attempt to copy its
    // contents.  Note that it is possible for both right and left sides to be
    // found terminal.  Such nodes will be ignored by restaging, for the same
    // reason that nodes found unsplitable during this interlevel pass are ignored,
    // i.e., by virtue of having both 'restage' fields set to negative values.
    //
    int lhSCount;
    splitSig->LHSizes(lhSCount, lhIdxCount);
    double minInfoNext = splitSig->MinInfo();

    if (TerminalSize(lhSCount, lhIdxCount))
      ptL = -1;  // Reset to flag terminality.  EXIT?
    else {
      lNext = lhSplitCount++;
      IndexNode::NextLevel(lNext, ptL, lhIdxCount, lhSCount, lhSum, minInfoNext);
    }
    if (TerminalSize(sCount - lhSCount, idxCount - lhIdxCount))
      ptR = -1; // Reset to flag terminality.  EXIT?
    else {
      rNext = lhSplitNext + rhSplitCount++;
      IndexNode::NextLevel(rNext, ptR, idxCount - lhIdxCount, sCount - lhSCount, sum - lhSum, minInfoNext);
    }
  }

  // Consumes all fields essential for restaging.
  //
  RestageMap::ConsumeSplit(this - nodeCache, lNext, rNext, lNext >= 0 ? lhIdxCount : 0, rNext >= 0 ? idxCount - lhIdxCount : 0);
}


// Transfers inter-level NodeCache contents to next level's split records (IndexNodes).
// Pre-bias computation is delayed, however, until all Replay activity is completed.
//
// N.B.:  Data cached in NodeCache nodes continue to record the state of the previous
// level's split nodes and remain accessible until overwritten by the next interlevel
// pass.  In particular, the cached nodes can guide restaging of the predictor splits
// from the previous level to the next level.
//
void IndexNode::NextLevel(int _splitIdx, int _ptId, int _idxCount, int _sCount, double _sum, double _minInfo) {
  IndexNode *idxNode = &indexNode[_splitIdx];
  idxNode->idxCount = _idxCount;
  idxNode->sCount = _sCount;
  idxNode->sum = _sum;
  idxNode->ptId = _ptId;
  idxNode->minInfo = _minInfo;
}

// Sets the "late" fields for use by the upcoming level.
//
// 'lhStart' could be set within NextLevel() but this would require maintaining
// much more internal state. 'preBias', in the categorical case, requires the
// level's state sums to be available, which in turn requires that all Replay()
// calls have completed.
//
void IndexNode::LateFields(int splitCount) {
  int start = 0;
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    IndexNode *idxNode = &indexNode[splitIdx];
    idxNode->preBias = Response::PrebiasSt(splitIdx);
    idxNode->lhStart = start;
    start += idxNode->idxCount; 
  }
}

// Two-sided Replay(), called for numeric SplitSigs, for which only the left-hand
// index count is known.  The right-hand count is derived by subtracting left-hand
// size from cached node's overall index count, 'idxCount'.
//
double NodeCache::ReplayNum(int splitIdx, int predIdx, int level, int lhIdxCount) {
  int ptLH, ptRH, start, end;
  RestageFields(splitIdx, ptLH, ptRH, start, end);

  double lhSum = SamplePred::Replay(predIdx, level, start, start + lhIdxCount - 1, ptLH);
  (void) SamplePred::Replay(predIdx, level, start + lhIdxCount, end, ptRH);

  return lhSum;
}
