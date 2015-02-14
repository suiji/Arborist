// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file index.cc

   @brief Methods for maintaining the index-tree representation of splitable nodes.

   @author Mark Seligman

 */

#include "index.h"
#include "splitsig.h"
#include "pretree.h"
#include "response.h"
#include "train.h"
#include "splitpred.h"
#include "samplepred.h"

// Testing only:
//#include <iostream>
using namespace std;


IndexNode *IndexNode::indexNode = 0;
int IndexNode::totLevels = -1;
int IndexNode::levelMax = -1;

int NodeCache::minHeight = -1;
NodeCache *NodeCache::nodeCache = 0;

/**
   @brief Lights off the necessary initializations.

   @param _minHeight is the minimum node size for splitting.

   @param _totLevels is an upper bound on level count.  Zero is reserved to indicate no limit.

   @return void.
 */
void IndexNode::Factory(int _minHeight, int _totLevels) {
  totLevels = _totLevels;

  NodeCache::Factory(_minHeight);
}

/**
   @brief Per-tree initialization of various members and dependent objects.

   @param _levelMax is the current level-max.

   @param _bagCount is the number of in-bag samples.

   @param nSamp is the number of samples from which this tree is built.

   @double sum is the sum of response values among the samples.
 */
void IndexNode::TreeInit(int _levelMax, int bagCount, int nSamp, double sum) {
  levelMax = _levelMax;
  indexNode = new IndexNode[levelMax];
  NodeCache::TreeInit();
  SplitPred::TreeInit();
  SplitSig::TreeInit(levelMax);
  NextLevel(0, 0, bagCount, nSamp, sum, 0.0);
}

/**
   @brief Allocates nodeCache[] with the current level-max.

   @return void.
 */
void NodeCache::TreeInit() {
  nodeCache = new NodeCache[levelMax];
}

/**
   @brief Deallocates the nodeCache[] vector.

   @return void.
 */
void NodeCache::TreeClear() {
  delete [] nodeCache;
  nodeCache = 0;
}

/**
   @brief Per-tree deallocations.

   @return void.
 */
void IndexNode::TreeClear() {
  NodeCache::TreeClear();
  delete [] indexNode;
  indexNode = 0;
}

/**
 @brief Updates 'levelMax' and data structures depending upon it.

 Invokes ReFactory() methods on only those classes which must be integral
 prior to NodeCache's consumption.

 @return void.
*/
void IndexNode::ReFactory() {
  levelMax = Train::ReFactory();

  delete [] indexNode;
  indexNode = new IndexNode[levelMax];

  // RestageMap:  restageMap[]
  RestageMap::ReFactory(levelMax);
}

/**
  @brief Reallocation of NodeCache, as well as classes not required for consumption of
  all instances.

  By delaying NodeCache reallocation until after consumption, content need not
  be copied.  This includes SplitSig references recorded directly on the cached
  node.

  @return void.
*/
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

/**
   @brief Finalization for class.

   @return void.
 */
void IndexNode::DeFactory() {
  delete [] indexNode;
  indexNode = 0;
  levelMax = -1;
  totLevels = -1;
  NodeCache::DeFactory();
}

/**
   @brief Records minimum node size.

   @return void.
 */
void NodeCache::Factory(int _minHeight) {
  minHeight = _minHeight;
}

/**
   @brief Class finalizer.

   @return void.
 */
void NodeCache::DeFactory() {
  minHeight = -1;
}

/**
   @brief Monolith entry point for per-level splitting.

   @return count of levels built.
*/
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

/**
 @brief Caches all indexNode[] elements from the current level into nodeCache[]
 workspace.

 By caching, the next level's index nodes can be populated without incurring crosstalk.

 @param splitCount is the count of index nodes to cache.

 @return void.
*/
void NodeCache::CacheNodes(int splitCount) {
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    Cache(splitIdx);
  }
}

/**
   @brief Performs the inter-level arbitration needed to initilialize split nodes for
   the next level.

   @param level is the current zero-based level.

   @param splitCount is the number of index nodes in the current level.

   @param lhSplitNext outputs the number of LH index nodes in the next level.

   @param rhSplitNext outputs the number of RH index nodes in the next level.

   @param leafNext outputs number of pretree terminals in the next level.

   @return Start of RH position for restaging, plus output parameters.

   TODO:  MUST guarantee that no zero-length "splits" have been introduced.
   Not only are these nonsensical, but they are also dangerous, as they violate
   various assumptions about the integrity of the intermediate respresentation.
*/

int NodeCache::InterLevel(int level, int splitCount, int &lhSplitNext, int &rhSplitNext, int &leafNext) {
  // The arg-max calls operate on distinct vectors.  Assuming that
  // # samples >> # predictors, parallelization of this loop
  // is warranted as 'splitCount' grows.
  //
  CacheNodes(splitCount);
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++)
    nodeCache[splitIdx].Splitable(level);

  // Restaging is implemented as a stable partition, and is facilitated by
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

/**
 @brief Ensures that sufficient space is present in the index tree and PreTree to
 accomodate the next level's nodes.

 @param splitCount is the number of index nodes in the current level.

 @param splitNext is the number of index nodes in the next level.

 @param leafNext is the number of pretree terminals in the next level.

 @return True iff IndexNode reallocation takes place.
*/
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

/**
   @brief Walks the list of cached splits from the level just concluded, adding PreTree
   terminals and IndexNodes for the next level.

   @param splitCount is the number of index nodes at this level.

   @param lhSplitNext is the number of LH index nodes in the next level.

   @param totLhIdx is the number of LH indices subsumed in the next level.

   @param reFac indicates whether reallocation has been found necessary.

   @return void.
*/
void NodeCache::NextLevel(int splitCount, int lhSplitNext, int totLhIdx, bool reFac) {
  PreTree::NextLevel();
  RestageMap::Commence(splitCount, totLhIdx);

  int lhCount = 0;
  int rhCount = 0;
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++)
    nodeCache[splitIdx].Consume(lhSplitNext, lhCount, rhCount);

  //  if (lhSplitNext != lhCount) // ASSERTION
	//cout << "Next level split-count mismatch" << endl;

  // Reallocates vectors potentially referenced by objects in the node cache.
  //
  if (reFac)
    ReFactory();
}

/**
 @brief Finds the maximal splitting predictor for this node and marks pretree accordingly.
 @param level is the current level number.

 @return void.
*/
void NodeCache::Splitable(int level) {
  int splitIdx = this - nodeCache;
  splitSig = SplitSig::ArgMax(splitIdx, level, preBias, minInfo);
  if (splitSig != 0) // Tags PreTree node as splitable.
    splitSig->NonTerminal(splitIdx, level,  ptId, lhStart);
}

/**
   @brief If the cached node splits, then a census is taken of the next level's
   left and right split nodes nodes and leaves.

   @param lhSplitNext outputs count of LH index nodes in next level.

   @param rhSplitNext outputs count of RH index nodes in next level.

   @param leafNext outputs count of pretree terminals in next level.

   @return count of indices subsumed by LH, plus output reference parameters.
*/
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

/**
   @brief Consumes all cached information for this node, following which the node should be considered dead.
   
   LH and RH pre-tree nodes are made for all split nodes actually found to be
   splitable during this interlevel pass.  Node indices for both sides are passed
   to the Replay().  Terminality constraints are checked and index tree nodes
   (IndexNode) are made for all sides not so constrained.  Split node order is
   assigned so as to correspond with expectations of restaging, which takes place
   at the start of the next level.

   @param lhSplitNext is the total number of LH index nodes in the next level.

   @param lhSplitCount outputs the accumulated number of next-level LH index nodes.

   @param rhSplitCount outputs the accumulated number of next-level RH index nodes.

   @return void, plus output reference parameters.
*/
void NodeCache::Consume(int lhSplitNext, int &lhSplitCount, int &rhSplitCount) {
  int lhIdxCount = 0;
  int lNext = -1;
  int rNext = -1;
  int splitIdx = this - nodeCache;
  if (splitSig != 0) {
    // Pre-tree nodes (PreTree) for both sides are initialized as leaves (i.e.,
    // terminal), and are only later updated as splits if found to be splitable
    // during the next inter-level pass.
    //
    PreTree::TerminalOffspring(ptId, ptL, ptR);

    // With LH and RH PreTree indices known, the sample indices associated with
    // this split node can be looked up and remapped.  Replay() assigns actual
    // index values, irrespective of whether the pre-tree nodes at these indices
    // are terminal or non-terminal.
    //
    double lhSum = splitSig->Replay(splitIdx, ptL, ptR);

    // Index tree nodes (IndexNode), OTOH, are only made for those sides with the
    // potential to split - that is, which are not already known to be terminal.
    // These are made by NextLevel(), which is invoked in a manner guaranteeing
    // an ordering in which left-hand splits precede right-hand splits.
    // This ordering ensures that offset values assigned to splits reflect the same
    // ordering as is assigned by restaging, which effects a stable partition of
    // this level's predictor sample orderings (SampleOrd).
    //
    int lhSCount;
    splitSig->LHSizes(lhSCount, lhIdxCount);
    double minInfoNext = splitSig->MinInfo();

    if (!TerminalSize(lhSCount, lhIdxCount)) {
      lNext = lhSplitCount++;
      IndexNode::NextLevel(lNext, ptL, lhIdxCount, lhSCount, lhSum, minInfoNext);
    }
    if (!TerminalSize(sCount - lhSCount, idxCount - lhIdxCount)) {
      rNext = lhSplitNext + rhSplitCount++;
      IndexNode::NextLevel(rNext, ptR, idxCount - lhIdxCount, sCount - lhSCount, sum - lhSum, minInfoNext);
    }
  }

  // Consumes all fields essential for restaging.
  //
  RestageMap::ConsumeSplit(splitIdx, lNext, rNext, lNext >= 0 ? lhIdxCount : 0, rNext >= 0 ? idxCount - lhIdxCount : 0);
}

/**
   @brief Transfers majority of inter-level NodeCache contents to next level's split records (IndexNodes).
   @see LateFields

   @param _splitIdx is the index of the split referenced.

   @param _ptId is the pretree node index.

   @param _idxCount is the count indices represented.

   @param _sCount is the count of samples represented.

   @param _sum is the sum of response values at the indices represented.

   @param _minInfo is the minimal information content suitable to split either child.

   @return void.
*/
void IndexNode::NextLevel(int _splitIdx, int _ptId, int _idxCount, int _sCount, double _sum, double _minInfo) {
  IndexNode *idxNode = &indexNode[_splitIdx];
  idxNode->idxCount = _idxCount;
  idxNode->sCount = _sCount;
  idxNode->sum = _sum;
  idxNode->ptId = _ptId;
  idxNode->minInfo = _minInfo;
}

/**
   @brief Sets the "late" fields for use by the upcoming level.

   'lhStart' could be set within NextLevel() but this would require maintaining
   much more internal state. 'preBias', in the categorical case, requires the
   level's state sums to be available, which in turn requires that all Replay()
   calls have completed.

   @param splitCount is the number of index nodes in the (next) level.

   @return void.
*/
void IndexNode::LateFields(int splitCount) {
  int start = 0;
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    IndexNode *idxNode = &indexNode[splitIdx];
    idxNode->preBias = Response::PrebiasSt(splitIdx);
    idxNode->lhStart = start;
    start += idxNode->idxCount; 
  }
}

/**
   @brief Two-sided Replay(), called for numeric SplitSigs, for which only the left-hand
   index count is known.

  The right-hand count is derived by subtracting left-hand size from cached node's
  overall index count, 'idxCount'.

  @param splitIdx is the index of the split referenced.

  @param predIdx is the index of the splitting predictor.

  @param level is the current level.

  @param lhIdxCount is the total number of indices referenced by the left-hand node.

  @return sum of response values for the left-hand side.
*/
double NodeCache::ReplayNum(int splitIdx, int predIdx, int level, int lhIdxCount) {
  int ptLH, ptRH, start, end;
  RestageFields(splitIdx, ptLH, ptRH, start, end);

  double lhSum = SamplePred::Replay(predIdx, level, start, start + lhIdxCount - 1, ptLH);
  (void) SamplePred::Replay(predIdx, level, start + lhIdxCount, end, ptRH);

  return lhSum;
}
