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
#include "bv.h"
#include "pretree.h"
#include "sample.h"
#include "splitsig.h"
#include "samplepred.h"
#include "bottom.h"

// Testing only:
//#include <iostream>
//using namespace std;
//#include <time.h>
//clock_t clock(void);


unsigned int Index::totLevels = 0;

/**
   @brief Initialization of static invariants.

   @param _minNode is the minimum node size for splitting.

   @param _totLevels is the maximum number of levels to evaluate.

   @return void.
 */
void Index::Immutables(unsigned int _minNode, unsigned int _totLevels) {
  NodeCache::Immutables(_minNode);
  totLevels = _totLevels;
}


/**
   @brief Reset of statics.

   @return void.
 */
void Index::DeImmutables() {
  totLevels = 0;
  NodeCache::DeImmutables();
}


unsigned int NodeCache::minNode = 0;

void NodeCache::Immutables(unsigned int _minNode) {
  minNode = _minNode;
}


void NodeCache::DeImmutables() {
  minNode = 0;
}


/**
   @brief Per-tree constructor.  Sets up root node for level zero.
 */
Index::Index(SamplePred *_samplePred, PreTree *_preTree, Bottom *_bottom, int _nSamp, int _bagCount, double _sum) : bagCount(_bagCount), samplePred(_samplePred), preTree(_preTree), bottom(_bottom) {
  levelBase = 0;
  levelWidth = 1;
  indexNode = new IndexNode[1];
  indexNode[0].Init(0, 0, 0, _bagCount, _nSamp, _sum, 0.0, 0);
}


/**
   @brief Destructor.

   @return void.
 */
Index::~Index() {
}


IndexNode::IndexNode() : splitIdx(0), lhStart(0), idxCount(0), sCount(0), sum(0.0), minInfo(0.0), ptId(0), path(0) {
}

NodeCache::NodeCache() : IndexNode(), terminal(true) {}

/**
   @brief Instantiates a block of PreTees for bulk return, but may or may
   not build them concurrently.

   @param sampleBlock contains the sample objects characterizing the roots.

   @param treeBlock is the number of trees to train in this block.

   @return brace of 'treeBlock'-many PreTree objects.
*/
PreTree **Index::BlockTrees(Sample **sampleBlock, int treeBlock) {
  PreTree **ptBlock = new PreTree*[treeBlock];

  for (int blockIdx = 0; blockIdx < treeBlock; blockIdx ++) {
    Sample *sample = sampleBlock[blockIdx];
    ptBlock[blockIdx] = OneTree(sample->SmpPred(), sample->Bot(), Sample::NSamp(), sample->BagCount(), sample->BagSum());
  }
  
  return ptBlock;
}


/**
   @brief Performs sampling and level processing for a single tree.

   @return void.
 */
PreTree *Index::OneTree(SamplePred *_samplePred, Bottom *_bottom, int _nSamp, int _bagCount, double _sum) {
  PreTree *_preTree = new PreTree(_bagCount);
  Index *index = new Index(_samplePred, _preTree, _bottom, _nSamp, _bagCount, _sum);
  index->Levels();
  delete index;

  return _preTree;
}


/**
   @brief Main loop for per-level splitting.  Assumes root node and attendant per-tree
   data structures have been initialized.

   @return void.
*/
void  Index::Levels() {
  unsigned int levelCount = 1;
  for (unsigned int level = 0; levelCount > 0; level++) {
    bottom->LevelInit();
    unsigned int splitNext, lhNext, leafNext;
    NodeCache *nodeCache = LevelConsume(levelCount, splitNext, lhNext, leafNext);
    if (splitNext != 0 && level + 1 != totLevels) {
      LevelProduce(nodeCache, level, levelCount, splitNext, lhNext, leafNext);
      levelCount = splitNext;
    }
    else {
      levelCount = 0;
    }
    delete [] nodeCache;
    bottom->LevelClear();
  }
}
//  ASSERTION:
//   levelBase + levelWidth == preTree->TreeHeight()


/**
   @brief Consumes and caches previous level's nodes.

   @param argMax is a vector of split signatures corresponding to the
   nodes.

   @return a vector of cached node copies.
 */
NodeCache *Index::CacheNodes(const std::vector<SSNode*> &argMax) {
  NodeCache *nodeCache = new NodeCache[argMax.size()];
  for (unsigned int splitIdx = 0; splitIdx < argMax.size(); splitIdx++) {
    nodeCache[splitIdx].Cache(&indexNode[splitIdx], argMax[splitIdx]);
  }
  delete [] indexNode;

  return nodeCache;
}


/**
   @brief Counts splits and leaves in the next level.

   @lhSplitNext outputs the count of left-hand splits in the next level.

   @leafNext outputs the number of leaves in the next level.

   @return total number of splits in the next level.
 */
unsigned int Index::LevelCensus(NodeCache nodeCache[], unsigned int levelCount, unsigned int &lhSplitNext, unsigned int &leafNext) {
  lhSplitNext = leafNext = 0;
  unsigned int rhSplitNext = 0;
  for (unsigned int splitIdx = 0; splitIdx < levelCount; splitIdx++)
    nodeCache[splitIdx].SplitCensus(lhSplitNext, rhSplitNext, leafNext);
  
  // Restaging is implemented as a patient stable partition.
  //
  // Coprocessor implementations can be streamlined using a node-
  // independent indexing scheme, e.g., enumerating all left-hand
  // subnodes before the first right-hand subnode.
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
void NodeCache::SplitCensus(unsigned int &lhSplitNext, unsigned int &rhSplitNext, unsigned int &leafNext) {
  if (ssNode == 0)
    return;

  ssNode->LHSizes(lhSCount, lhIdxCount);
  if (Splitable(lhIdxCount)) {
    lhSplitNext++;
  }
  else {
    leafNext++;
  }

  if (Splitable(idxCount - lhIdxCount)) {
    rhSplitNext++;
  }
  else
    leafNext++;
}


/**
   @brief Walks the list of split signatures for the level just concluded,
   adding pre-tree and Index nodes for the next level.

   @param level is the current level.

   @return count of nodes at next level:  zero if short-circuiting.
*/
NodeCache *Index::LevelConsume(unsigned int levelCount, unsigned int &splitNext, unsigned int &lhSplitNext, unsigned int &leafNext) {
  NodeCache *nodeCache = CacheNodes(bottom->Split(this, indexNode));
  splitNext = LevelCensus(nodeCache, levelCount, lhSplitNext, leafNext);

  // Next level of pre-tree needs sufficient space to consume splits
  // precipitated by cached nodes.
  preTree->CheckStorage(splitNext, leafNext);
  for (unsigned int splitIdx = 0; splitIdx < levelCount; splitIdx++) {
    nodeCache[splitIdx].Consume(preTree, samplePred, bottom);
  }

  return nodeCache;
}


void Index::LevelProduce(NodeCache *nodeCache, unsigned int level, unsigned int levelCount, unsigned int splitNext, unsigned int lhSplitNext, unsigned int leafNext) {
  levelBase += levelWidth;
  levelWidth = splitNext + leafNext;

  ntLH = new bool[levelWidth];
  ntRH = new bool[levelWidth];
  for (unsigned int i = 0; i < levelWidth; i++)
    ntLH[i] = ntRH[i] = false;

  unsigned int lhCount = 0;
  unsigned int rhCount = 0;

  // Next call guaranteed, so no dangling references:
  indexNode = new IndexNode[splitNext];
  bottom->NewLevel(splitNext);
  for (unsigned int splitIdx = 0; splitIdx < levelCount; splitIdx++) {
    nodeCache[splitIdx].Successors(this, preTree, samplePred, bottom, lhSplitNext, lhCount, rhCount);
  }
  LRLive(bottom, nodeCache, level);

  // Assigns start values to consecutive nodes at next level.
  /*
  unsigned int idxCount = 0;
  for (unsigned int splitIdx = 0; splitIdx < splitNext; splitIdx++) {
    indexNode[splitIdx].Start() = idxCount;
    idxCount += indexNode[splitIdx].IdxCount();
  }
  */

  delete [] ntLH;
  delete [] ntRH;
  ntLH = ntRH = 0;
}


/**
  @brief Consumes all remaining information for node in the current level.

  @return void.
*/
void NodeCache::Consume(PreTree *preTree, SamplePred *samplePred, Bottom *bottom) {
  if (ssNode != 0) {
    lhSum = ssNode->NonTerminal(samplePred, preTree, splitIdx, lhStart, lhStart + idxCount - 1, ptId, ptL, ptR, bottom->Runs());
  }
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
void NodeCache::Successors(Index *index, PreTree *preTree, SamplePred *samplePred, Bottom *bottom, unsigned int lhSplitNext, unsigned int &lhSplitCount, unsigned int &rhSplitCount) {
  if (ssNode != 0) {
    if (Splitable(lhIdxCount)) {
      terminal = false;
      unsigned int lNext = lhSplitCount++;
      unsigned int start = lhStart;
      unsigned int pathNext = index->NextLH(lNext, ptL, start, lhIdxCount, lhSCount, lhSum, ssNode->MinInfo(), path);
      bottom->ReachingPath(splitIdx, pathNext, lNext, start, lhIdxCount);
    }

    if (Splitable(idxCount - lhIdxCount)) {
      terminal = false;
      unsigned int rNext = lhSplitNext + rhSplitCount++;
      unsigned int start = lhStart + lhIdxCount;
      unsigned int pathNext = index->NextRH(rNext, ptR, start, idxCount - lhIdxCount, sCount - lhSCount, sum - lhSum, ssNode->MinInfo(), path);
      bottom->ReachingPath(splitIdx, pathNext, rNext, start, idxCount - lhIdxCount);
    }
  }
}


/**
   @brief Packs live lh/rh information into bit vectors and zero-pads up to the next slot boundary.

   @param bitsLH[] outputs live LH bits.

   @param bitsRH[] outputs live RH bits.

   @param lhIdxTot outputs the count of LH indices used in the upcoming level.

   @param rhIdxTot outputs the count of RH indices used in the upcoming level.

   @return void, with output parameters.
 */
void Index::PredicateBits(BV *sIdxLH, BV *sIdxRH, int &lhIdxTot, int &rhIdxTot) const {
  lhIdxTot = rhIdxTot = 0;
  unsigned int slotBits = BV::SlotElts();
  int slot = 0;
  for (unsigned int base = 0; base < bagCount; base += slotBits, slot++) {
    unsigned int lhBits = 0;
    unsigned int rhBits = 0;
    unsigned int mask = 1;
    unsigned int supIdx = bagCount < base + slotBits ? bagCount : base + slotBits;
    for (unsigned int sIdx = base; sIdx < supIdx; sIdx++, mask <<= 1) {
      unsigned int levelOff;
      bool atLevel = LevelOffSample(sIdx, levelOff);
      if (atLevel) {
	bool isLH = ntLH[levelOff];
	lhIdxTot += isLH ? 1 : 0;
	lhBits |= isLH ? mask : 0;
	bool isRH = ntRH[levelOff];
	rhIdxTot += isRH ? 1 : 0;
	rhBits |= ntRH[levelOff] ? mask : 0;
      }
    }
    sIdxLH->SetSlot(slot, lhBits);
    sIdxRH->SetSlot(slot, rhBits);
  }
}


/**
   @brief 
 */
void Index::LRLive(Bottom *bottom, const NodeCache *nodeCache, unsigned int level) const {
  for (unsigned int sIdx = 0; sIdx < bagCount; sIdx++) {
    unsigned int levelOff; // Not dense w.r.t. levelIdx.
    if (LevelOffSample(sIdx, levelOff)) {
      if (ntLH[levelOff])
	bottom->PathLeft(sIdx);
      else if (ntRH[levelOff])
	bottom->PathRight(sIdx);
      else {
	bottom->PathExtinct(sIdx);
      }
    }
    else
      bottom->PathExtinct(sIdx);
  }
}


/**
   @brief Consults pretree for node holding sample and computes node's offset.

   @param sIdx is the sample index.

   @param levelOff is the level-relative offset of node holding sample index, if any.

   @return true iff node holding sample is at current level.
 */
bool Index::LevelOffSample(unsigned int sIdx, unsigned int &levelOff) const {
  unsigned int ptIdx = preTree->Sample2Frontier(sIdx);
  if (ptIdx >= levelBase) {
    levelOff = ptIdx - levelBase;
    // ASSERTION:  levelOff <= levelWidth;

    return true;
  }
  else {
    levelOff = 0; // dummy value.
    return false;
  }
}
