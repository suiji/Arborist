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
#include "splitpred.h"
#include "samplepred.h"
#include "restage.h"

// Testing only:
#include <iostream>
using namespace std;

int Index::totLevels = -1;
int Index::nSamp = -1;

/**
   @brief Initialization of static invariants.

   @param _minHeight is the minimum node size for splitting.

   @param _totLevels is the maximum number of levels to evaluate.

   @param _nSamp is the total sample count.  Only used to initialize root.

   @return void.
 */
void Index::Immutables(int _minHeight, int _totLevels, int _nSamp) {
  NodeCache::Immutables(_minHeight);
  totLevels = _totLevels;
  nSamp = _nSamp;
}


/**
   @brief Reset of statics.

   @return void.
 */
void Index::DeImmutables() {
  totLevels = nSamp = -1;
  NodeCache::DeImmutables();
}


int NodeCache::minHeight = -1;

void NodeCache::Immutables(int _minHeight) {
  minHeight = _minHeight;
}


void NodeCache::DeImmutables() {
  minHeight = -1;
}


/**
   @brief Per-tree constructor.  Sets up root node for level zero.
 */
Index::Index() {
  levelBase = 0;
  levelWidth = 1;
  splitCount = 1;
  indexNode = new IndexNode[1];
}


/**
   @brief Destructor.  There should always be an exposed indexNode vector
   to delete.

   @return void.
 */
Index::~Index() {
  delete [] indexNode;
}


/**
   @brief Instantiates a block of PreTees for bulk return, but may or may
   not build them concurrently.

   @param treeBlock is the number of trees to train in this block.

   @return brace of 'treeBlock'-many PreTree objects.
*/
PreTree **Index::BlockTrees(const PredOrd *predOrd, int treeBlock) {
  PreTree **ptBlock = new PreTree*[treeBlock];

  int levelBlock = 1;  // For now, only building sequentially.
  for (int treeIdx = 0; treeIdx < treeBlock; treeIdx += levelBlock) {
    Index *index = new Index();
    ptBlock[treeIdx] = index->Root(predOrd);
    index->Levels();
    delete index;
  }

  return ptBlock;
}


/**
   @brief Initializes root node and fires off attendant classes' per-tree methods.

   @param predOrd is the sorted predictor table.

   @return Reference to current PreTree.
 */
PreTree *Index::Root(const PredOrd *predOrd) {
  double sum;
  preTree = new PreTree();
  bagCount = preTree->BagRows(predOrd, samplePred, splitPred, sum);
  indexNode[0].Init(0, 0, bagCount, nSamp, sum, 0.0);

  return preTree;
}


/**
   @brief Main loop for per-level splitting.  Assumes root node and attendant per-tree
   data structures have been initialized.

   @return void.
*/
void  Index::Levels() {
  SplitSig *splitSig = new SplitSig();

  for (level = 0; splitCount > 0 && (totLevels == 0 || level < totLevels); level++) {
    splitSig->LevelInit(splitCount);
    splitPred->LevelInit(this, splitCount);
    splitPred->LevelSplit(indexNode, level, splitCount, splitSig);
    NodeCache *nodeCache = CacheNodes();
    ArgMax(nodeCache, splitSig);
    int lhSplitNext, leafNext;
    int splitNext = LevelCensus(nodeCache, lhSplitNext, leafNext);
    ProduceNext(nodeCache, splitNext, lhSplitNext, leafNext, level);
    splitSig->LevelClear();
    splitCount = splitNext;
  }

  delete samplePred;
  delete splitPred;
  delete splitSig;
  // ASSERTION:
  //   levelBase + levelWidth == preTree->TreeHeight()
}


/**
   @brief Sets (Gini) pre-bias value according to response type.

   @param indexNode is the index tree vector for the current level.

   @return void.
*/
void Index::SetPrebias() {
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    IndexNode *idxNode = &indexNode[splitIdx];
    int sCount;
    double sum;
    idxNode->PrebiasFields(sCount, sum);
    idxNode->Prebias() = splitPred->Prebias(splitIdx, sCount, sum);
  }
}


/**
   @brief Caches all indexNode[] elements from the current level into nodeCache[]
   workspace.

   By caching the current level's index nodes, the next level's nodes can be populated without incurring crosstalk.

   @param splitCount is the count of index nodes to cache.

 @return void.
*/
NodeCache *Index::CacheNodes() {
  NodeCache *nodeCache = new NodeCache[splitCount]; // Lives until consumption.
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    nodeCache[splitIdx].Cache(&indexNode[splitIdx]);
  }

  return nodeCache;
}


/**
  @brief  Walks level's split signatures to find maximal information content.

  Levels can grow quite wide, so parallelization is probably worthwhile.

  @param nodeCache is the vector of cached Index nodes for this level.

  @param _level is the current level.

  @return void.
*/
void Index::ArgMax(NodeCache nodeCache[], const SplitSig *splitSig) {
  int splitIdx;

#pragma omp parallel default(shared) private(splitIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < splitCount; splitIdx++) {
      nodeCache[splitIdx].SS() = splitSig->ArgMax(splitIdx, nodeCache[splitIdx].MinInfo());
    }
  }
}


/**
   @brief Counts splits and leaves in the next level.

   @lhSplitNext outputs the count of left-hand splits in the next level.

   @leafNext outputs the number of leaves in the next level.

   @return total number of splits in the next level.
 */
int Index::LevelCensus(NodeCache nodeCache[], int &lhSplitNext, int &leafNext) {
  lhSplitNext = leafNext = 0;
  int rhSplitNext = 0;
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++)
    nodeCache[splitIdx].SplitCensus(lhSplitNext, rhSplitNext, leafNext);
  
  // Restaging is implemented as a stable partition, and is facilitated by
  // enumerating all left-hand subnodes before the first right-hand subnode.
  // Whence the separate enumeration of the two.
  //
  return lhSplitNext + rhSplitNext;
}


/**
   @brief Splitable nodes only:  takes census next level's left, right split
   nodes nodes and leaves.

   @param lhSplitNext outputs count of LH index nodes in next level.

   @param rhSplitNext outputs count of RH index nodes in next level.

   @param leafNext outputs count of pretree terminals in next level.

   @return void, plus output reference parameters.
*/
void NodeCache::SplitCensus(int &lhSplitNext, int &rhSplitNext, int &leafNext) const {
  if (ssNode == 0)
    return;
  
  int lhSCount, lhIdxCount;
  ssNode->LHSizes(lhSCount, lhIdxCount);

  if (TerminalSize(lhSCount, lhIdxCount)) {
    leafNext++;
  }
  else {
    lhSplitNext++;
  }

  if (TerminalSize(sCount - lhSCount, idxCount - lhIdxCount)) {
    leafNext++;
  }
  else
    rhSplitNext++;
}


/**
   @brief Walks the list of cached splits from the level just concluded, adding PreTree
   terminals and IndexNodes for the next level.

   @param splitCount is the number of index nodes at this level.

   @param lhSplitNext is the number of LH index nodes in the next level.

   @param leafNext is the leaf count for the next level.

   @param level is the current level.

   @return void.
*/
void Index::ProduceNext(NodeCache *nodeCache, int splitNext, int lhSplitNext, int leafNext, int level) {
  // Next level of pre-tree needs sufficient space to consume
  // splits precipitated by cached nodes.
  preTree->CheckStorage(splitNext, leafNext);

  levelBase += levelWidth;
  levelWidth = splitNext + leafNext;

  delete [] indexNode;
  indexNode = new IndexNode[splitNext];

  ntLH = new bool[levelWidth];
  ntRH = new bool[levelWidth];
  for (unsigned int i = 0; i < levelWidth; i++)
    ntLH[i] = ntRH[i] = false;

  int lhCount = 0;
  int rhCount = 0;
  RestageMap *restageMap = new RestageMap(splitPred, bagCount, splitCount, splitNext);
  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++)
    nodeCache[splitIdx].Consume(this, preTree, splitPred, samplePred, restageMap, level, lhSplitNext, lhCount, rhCount);
  delete [] nodeCache;

  // SplitPred frees FacRun structures, which must persist until SplitSigs consumed.
  splitPred->LevelClear();

  // Assigns start values to consecutive nodes.
  //
  int idxCount = 0;
  for (int splitIdx = 0; splitIdx < splitNext; splitIdx++) {
    indexNode[splitIdx].Start() = idxCount;
    idxCount += indexNode[splitIdx].IdxCount();
  }

  restageMap->Conclude(this);
  if (splitNext > 0 && level + 1 != totLevels)
    restageMap->RestageLevel(samplePred, level+1);
  delete restageMap;

  delete [] ntLH;
  delete [] ntRH;
  ntLH = ntRH = 0;

  // Destructor cleans up exposed indexNode.
}


/**
   @brief Invoked from the RHS or LHS of a split to determine whether the node persists to the next.
   
   MUST guarantee that no zero-length "splits" have been introduced.
   Not only are these nonsensical, but they are also dangerous, as they violate
   various assumptions about the integrity of the intermediate respresentation.

   @param _SCount is the count of samples subsumed by the node.

   @param _idxCount is the count of indices subsumed by the node.

   @return true iff the node subsumes too few samples or is representable as a
     single buffer element.
*/
inline bool NodeCache::TerminalSize(int _sCount, int _idxCount) {
  return (_sCount < minHeight) || (_idxCount <= 1);
}


/**
   @brief Consumes all cached information for this node, following which the node should be considered dead.
   
   LH and RH pre-tree nodes are made for all split nodes actually found to be
   splitable during this interlevel pass.  Node indices for both sides are passed
   to the Replay().  Terminality constraints are checked and index tree nodes
   (IndexNode) are made for all sides not so constrained.  Split node order is
   assigned so as to correspond with expectations of restaging, which takes place
   at the start of the next level.

   Index tree nodes (IndexNode), OTOH, are only made for those sides with the
   potential to split - that is, which are not already known to be terminal.
   These are made by NextLevel(), which is invoked in a manner guaranteeing
   an ordering in which left-hand splits precede right-hand splits.
   This ordering ensures that offset values assigned to splits reflect the same
   ordering as is assigned by restaging, which effects a stable partition of
   this level's predictor sample orderings (SampleOrd).

   @param lhSplitNext is the total number of LH index nodes in the next level.

   @param lhSplitCount outputs the accumulated number of next-level LH index nodes.

   @param rhSplitCount outputs the accumulated number of next-level RH index nodes.

   @return void, plus output reference parameters.
*/
void NodeCache::Consume(Index *index, PreTree *preTree, SplitPred *splitPred, SamplePred *samplePred, RestageMap *restageMap, int level, int lhSplitNext, int &lhSplitCount, int &rhSplitCount) {
  int lhIdxCount = 0;
  int lNext = -1;
  int rNext = -1;
  if (ssNode != 0) {
    double lhSum = ssNode->NonTerminal(samplePred, preTree, splitPred, level, lhStart, lhStart + idxCount - 1, ptId, ptL, ptR);
    int lhSCount;
    ssNode->LHSizes(lhSCount, lhIdxCount);
    double minInfoNext = ssNode->MinInfo();

    if (!TerminalSize(lhSCount, lhIdxCount)) {
      lNext = lhSplitCount++;
      index->NextLH(lNext, ptL, lhIdxCount, lhSCount, lhSum, minInfoNext);
    }
    if (!TerminalSize(sCount - lhSCount, idxCount - lhIdxCount)) {
      rNext = lhSplitNext + rhSplitCount++;
      index->NextRH(rNext, ptR, idxCount - lhIdxCount, sCount - lhSCount, sum - lhSum, minInfoNext);
    }
  }

  // Consumes all fields essential for restaging.
  //
  restageMap->ConsumeSplit(splitIdx, lNext, rNext, lhIdxCount, idxCount - lhIdxCount, lhStart, lhStart + idxCount - 1);
}


/**
   @brief Packs live lh/rh information into bit vectors and zero-pads up to the next slot boundary.

   @param bitsLH[] outputs live LH bits.

   @param bitsRH[] outputs live RH bits.

   @param lhIdxTot outputs the count of LH indices used in the upcoming level.

   @param rhIdxTot outputs the count of RH indices used in the upcoming level.

   @return void, with output parameters.
 */
void Index::PredicateBits(unsigned int sIdxLH[], unsigned int sIdxRH[], int &lhIdxTot, int &rhIdxTot) const {
  lhIdxTot = rhIdxTot = 0;
  const unsigned int slotBits = 8 * sizeof(unsigned int);
  int slot = 0;
  for (unsigned int base = 0; base < bagCount; base += slotBits, slot++) {
    unsigned int lhBits = 0;
    unsigned int rhBits = 0;
    unsigned int mask = 1;
    unsigned int supIdx = bagCount < base + slotBits ? bagCount : base + slotBits;
    for (unsigned int sIdx = base; sIdx < supIdx; sIdx++, mask <<= 1) {
      int levelOff = LevelOffSample(sIdx);
      if (levelOff >= 0) {
	bool isLH = ntLH[levelOff];
	lhIdxTot += isLH ? 1 : 0;
	lhBits |= isLH ? mask : 0;
	bool isRH = ntRH[levelOff];
	rhIdxTot += isRH ? 1 : 0;
	rhBits |= ntRH[levelOff] ? mask : 0;
      }
    }
    sIdxLH[slot] = lhBits;
    sIdxRH[slot] = rhBits;
  }
}


/**
   @brief Consults pretree for node holding sample and computes node's offset.

   @param sIdx is the sample index.

   @return level-relative offset of node holding sample index. 
 */
int Index::LevelOffSample(unsigned int sIdx) const {
  return preTree->Sample2Frontier(sIdx) - levelBase;
}
