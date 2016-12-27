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
Index::Index(SamplePred *_samplePred, PreTree *_preTree, Bottom *_bottom, int _nSamp, int _bagCount, double _sum) : indexNode(std::vector<IndexNode>(1)), bagCount(_bagCount), levelWidth(1), samplePred(_samplePred), preTree(_preTree), bottom(_bottom) {
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


NodeCache::NodeCache() : IndexNode() {}


/**
   @brief Instantiates a block of PreTees for bulk return, but may or may
   not build them concurrently.

   @param sampleBlock contains the sample objects characterizing the roots.

   @param treeBlock is the number of trees to train in this block.

   @return brace of 'treeBlock'-many PreTree objects.
*/
PreTree **Index::BlockTrees(const PMTrain *pmTrain, Sample **sampleBlock, int treeBlock) {
  PreTree **ptBlock = new PreTree*[treeBlock];

  for (int blockIdx = 0; blockIdx < treeBlock; blockIdx ++) {
    Sample *sample = sampleBlock[blockIdx];
    ptBlock[blockIdx] = OneTree(pmTrain, sample->SmpPred(), sample->Bot(), Sample::NSamp(), sample->BagCount(), sample->BagSum());
  }
  
  return ptBlock;
}


/**
   @brief Performs sampling and level processing for a single tree.

   @return void.
 */
PreTree *Index::OneTree(const PMTrain *pmTrain, SamplePred *_samplePred, Bottom *_bottom, int _nSamp, int _bagCount, double _sum) {
  PreTree *_preTree = new PreTree(pmTrain, _bagCount);
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
    //    cout << "\nLevel " << level << "\n" << endl;
    bottom->LevelInit();
    unsigned int leafNext, idxTot;
    NodeCache *nodeCache = LevelConsume(levelCount, splitNext, lhSplitNext, leafNext, idxTot);
    if (splitNext != 0 && level + 1 != totLevels) {
      bottom->Overlap(splitNext, idxTot);
      LevelProduce(nodeCache, level, levelCount, leafNext);
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

  return nodeCache;
}


/**
   @brief Counts splits and leaves in the next level.

   @lhSplitNext outputs the count of left-hand splits in the next level.

   @leafNext outputs the number of leaves in the next level.

   @return total number of splits in the next level.
 */
unsigned int Index::LevelCensus(NodeCache nodeCache[], unsigned int levelCount, unsigned int &lhSplitNext, unsigned int &leafNext, unsigned int &idxTot) {
  lhSplitNext = leafNext = idxTot = 0;
  unsigned int rhSplitNext = 0;
  for (unsigned int splitIdx = 0; splitIdx < levelCount; splitIdx++)
    nodeCache[splitIdx].SplitCensus(lhSplitNext, rhSplitNext, leafNext, idxTot);
  
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
void NodeCache::SplitCensus(unsigned int &lhSplitNext, unsigned int &rhSplitNext, unsigned int &leafNext, unsigned int &idxTot) {
  if (ssNode == 0) {
    return;
  }

  ssNode->LHSizes(lhSCount, lhIdxCount);
  if (Splitable(lhIdxCount)) {
    lhSplitNext++;
    idxTot += lhIdxCount;
  }
  else {
    leafNext++;
  }

  if (Splitable(idxCount - lhIdxCount)) {
    rhSplitNext++;
    idxTot += idxCount - lhIdxCount;
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
NodeCache *Index::LevelConsume(unsigned int levelCount, unsigned int &splitNext, unsigned int &lhSplitNext, unsigned int &leafNext, unsigned int &idxTot) {
  NodeCache *nodeCache = CacheNodes(bottom->Split(this, indexNode));
  splitNext = LevelCensus(nodeCache, levelCount, lhSplitNext, leafNext, idxTot);

  unsigned int heightPrev = preTree->NextLevel(splitNext, leafNext);
  for (unsigned int splitIdx = 0; splitIdx < levelCount; splitIdx++) {
    nodeCache[splitIdx].NonTerminal(preTree, samplePred, bottom);
  }
  preTree->Preplay(heightPrev);
  
  for (unsigned int splitIdx = 0; splitIdx < levelCount; splitIdx++) {
    nodeCache[splitIdx].Consume(preTree, samplePred, bottom);
  }

  return nodeCache;
}


/**
  @brief Consumes all remaining information for node in the current level.

  @return void.
*/
void NodeCache::NonTerminal(PreTree *preTree, SamplePred *samplePred, Bottom *bottom) {
  if (ssNode != 0) {
    ssNode->NonTerminal(samplePred, preTree, bottom->Runs(), ptId, ptL, ptR);
  }
}


/**
  @brief Consumes all remaining information for node in the current level.

  @return void.
*/
void NodeCache::Consume(PreTree *preTree, SamplePred *samplePred, Bottom *bottom) {
  if (ssNode != 0) {
    lhSum = ssNode->Replay(samplePred, preTree, bottom->Runs(), idxCount, sum, ptId, ptL, ptR);
  }
}


void Index::LevelProduce(NodeCache *nodeCache, unsigned int level, unsigned int splitPrev, unsigned int leafNext) {
  levelWidth = splitNext + leafNext;

  unsigned int lhCount = 0;
  unsigned int rhCount = 0;
  // Next call guaranteed, so no dangling references:
  std::vector<IndexNode> _indexNode(splitNext);
  indexNode = std::move(_indexNode);

  // Speculatively sets all new node indices to canonical non-splitting index.
  std::vector<unsigned int> _ntNext(levelWidth);
  std::fill(_ntNext.begin(), _ntNext.end(), splitNext);
  ntNext = std::move(_ntNext);

  for (unsigned int splitIdx = 0; splitIdx < splitPrev; splitIdx++) {
    nodeCache[splitIdx].Successors(this, preTree, bottom, lhSplitNext, lhCount, rhCount);
  }

  RelIdx();
}


/**
   @brief Enumerates base indices for each node extant in the upcoming
   level.

   @return void.
 */
void Index::RelIdx() {
  unsigned int splitNext = indexNode.size();
  std::vector<unsigned int> relIdx(splitNext + 1);
  unsigned int idxTot = 0;
  for (unsigned int splitIdx = 0; splitIdx < splitNext; splitIdx++) {
    relIdx[splitIdx] = idxTot;
    idxTot += indexNode[splitIdx].IdxCount();
  }
  relIdx[splitNext] = idxTot;
  
  bottom->PathUpdate(this, relIdx);
}


/**
     @brief Looks up index node associated with frontier node.

     @param sIdx is the sample index lookup key.

     @param indexNext outputs either an IndexNode index or a placeholder
     value, depending whether the frontier node is a nonterminal.

     @return true iff frontier node defined in current level.
*/
bool Index::IndexNext(unsigned int sIdx, unsigned int &indexNext) const {
  unsigned int levelOff;
  bool ofLevel = preTree->SampleOffset(sIdx, levelOff);

  indexNext = ofLevel ? ntNext[levelOff] : bagCount;
  return ofLevel;
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
void NodeCache::Successors(Index *index, PreTree *preTree, Bottom *bottom, unsigned int lhSplitNext, unsigned int &lhSplitCount, unsigned int &rhSplitCount) {
  if (ssNode == 0) {
    return;
  }

  if (Splitable(lhIdxCount)) {
    index->NodeNext(splitIdx, lhSplitCount++, ptL, lhStart, lhIdxCount, lhSCount, lhSum, ssNode->MinInfo(), index->PathLeft(path));
  }

  if (Splitable(idxCount - lhIdxCount)) {
    index->NodeNext(splitIdx, lhSplitNext + rhSplitCount++, ptR, lhStart + lhIdxCount, idxCount - lhIdxCount, sCount - lhSCount, sum - lhSum, ssNode->MinInfo(), index->PathRight(path));
  }
}


void Index::NodeNext(unsigned int parIdx, unsigned int idxNext, unsigned int ptId, unsigned int start, unsigned int idxCount, unsigned int sCount, double sum, double minInfo, unsigned pathNext) {
  indexNode[idxNext].Init(idxNext, start, ptId, idxCount, sCount, sum, minInfo, pathNext);
  NTNext(ptId, idxNext);
  bottom->ReachingPath(parIdx, pathNext, idxNext, start, idxCount);
}

  
/**
   @brief Maps (absolute) nonterminal index to (compressed) index at next level.

   @param ptId is the pretree index of a nonterminal node.

   @param idxNext is the IndexNode index at representing the node at the
   next level.

   @return void.
 */
void Index::NTNext(unsigned int ptId, unsigned int idxNext) {
  ntNext[preTree->LevelOffset(ptId)] = idxNext;
}


/**
   @brief Consults pretree for node holding sample and computes node's offset.

   @param sIdx is the sample index.

   @param levelOff is the level-relative offset of node holding sample index, if any.

   @return true iff node holding sample is at current level.
 */
bool Index::LevelOffSample(unsigned int sIdx, unsigned int &levelOffset) const {
  return preTree->SampleOffset(sIdx, levelOffset);
}


/**
   @brief Returns the level-relative offset associated with an index node.

   @param splitIdx is the split index referenced.

   @return pretree offset from level base.
  */

unsigned int Index::LevelOffSplit(unsigned int splitIdx) const {
  return preTree->LevelOffset(indexNode[splitIdx].ptId);
}
