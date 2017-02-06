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
unsigned int IndexNode::minNode = 0;

/**
   @brief Initialization of static invariants.

   @param _minNode is the minimum node size for splitting.

   @param _totLevels is the maximum number of levels to evaluate.

   @return void.
 */
void Index::Immutables(unsigned int _minNode, unsigned int _totLevels) {
  IndexNode::minNode = _minNode;
  totLevels = _totLevels;
}


/**
   @brief Reset of statics.

   @return void.
 */
void Index::DeImmutables() {
  totLevels = 0;
  IndexNode::minNode = 0;
}



/**
   @brief Per-tree constructor.  Sets up root node for level zero.
 */
Index::Index(SamplePred *_samplePred, PreTree *_preTree, Bottom *_bottom, int _nSamp, unsigned int _bagCount, double _sum) : indexNode(std::vector<IndexNode>(1)), bagCount(_bagCount), levelWidth(1), samplePred(_samplePred), preTree(_preTree), bottom(_bottom) {
  indexNode[0].Init(0, 0, _nSamp, bagCount, 0, 0.0, 0, _sum, 0);
}


/**
   @brief Destructor.

   @return void.
 */
Index::~Index() {
}


IndexNode::IndexNode() : preBias(0.0), splitIdx(0), ptId(0), lhStart(0), idxCount(0), sCount(0), sum(0.0), minInfo(0.0), path(0), ssNode(0) {
}


/**
   @brief Instantiates a block of PreTees for bulk return, but may or may
   not build them concurrently.

   @param sampleBlock contains the sample objects characterizing the roots.

   @param treeBlock is the number of trees to train in this block.

   @return brace of 'treeBlock'-many PreTree objects.
*/
PreTree **Index::BlockTrees(const PMTrain *pmTrain, Sample **sampleBlock, int treeBlock) {
  PreTree **ptBlock = new PreTree*[treeBlock];

  for (int blockIdx = 0; blockIdx < treeBlock; blockIdx++) {
    Sample *sample = sampleBlock[blockIdx];
    ptBlock[blockIdx] = OneTree(pmTrain, sample->SmpPred(), sample->Bot(), Sample::NSamp(), sample->BagCount(), sample->BagSum());
  }
  
  return ptBlock;
}


/**
   @brief Performs sampling and level processing for a single tree.

   @return void.
 */
PreTree *Index::OneTree(const PMTrain *pmTrain, SamplePred *_samplePred, Bottom *_bottom, int _nSamp, unsigned int _bagCount, double _sum) {
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
  for (unsigned int level = 0; !indexNode.empty(); level++) {
    //cout << "\nLevel " << level << "\n" << endl;
    bottom->LevelInit();
    unsigned int lhSplitNext, leafNext, idxLive, idxMax;
    unsigned int splitNext = LevelCensus(bottom->Split(this, indexNode), lhSplitNext, leafNext, idxLive, idxMax);
    LevelConsume(splitNext, leafNext);
    if (splitNext != 0 && level + 1 != totLevels) {
      levelWidth = splitNext + leafNext;
      LevelProduce(bottom, splitNext, lhSplitNext, idxLive, idxMax);
    }
    else {
      indexNode.clear();
    }

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
unsigned int Index::LevelCensus(const std::vector<SSNode*> &argMax, unsigned int &lhSplitNext, unsigned int &leafNext, unsigned int &idxLive, unsigned int &idxMax) {
  unsigned int rhSplitNext, leafThis;
  rhSplitNext = leafThis = lhSplitNext = idxLive = idxMax = 0;

  unsigned int splitIdx = 0;
  for (auto & node : indexNode) {
    node.SplitCensus(argMax[splitIdx++], leafThis, lhSplitNext, rhSplitNext, idxLive, idxMax);
  }

  // Restaging is implemented as a patient stable partition.
  //
  // Coprocessor implementations can be streamlined using a node-
  // independent indexing scheme, e.g., enumerating all left-hand
  // subnodes before the first right-hand subnode.
  //
  unsigned int splitNext = lhSplitNext + rhSplitNext;
  leafNext = 2 * (indexNode.size() - leafThis) - splitNext;

  return splitNext;
}


/**
   @brief Consumes relevant contents of split signature, if any, and accumulates
   leaf and splitting census.
 */
void IndexNode::SplitCensus(SSNode *argMax, unsigned int &leafThis, unsigned int &lhSplitNext, unsigned int &rhSplitNext, unsigned int &idxLive, unsigned int &idxMax) {
  ssNode = argMax;
  if (ssNode == 0) {
    leafThis++;
  }
  else {
    ssNode->LHSizes(lhSCount, lhIdxCount);
    lhSplitNext += SplitAccum(lhIdxCount, idxLive, idxMax) ? 1 : 0;
    rhSplitNext += SplitAccum(idxCount - lhIdxCount, idxLive, idxMax) ? 1 : 0;
  }
}


/**
   @brief Walks the list of split signatures for the level just concluded,
   adding pre-tree and Index nodes for the next level.

   @param level is the current level.

   @return void.
*/
void Index::LevelConsume(unsigned int splitNext, unsigned int leafNext) {
  unsigned int heightPrev = preTree->Level(splitNext, leafNext);

  for (auto  & node : indexNode) {
    node.NonTerminal(preTree, samplePred, bottom);
  }
  preTree->Preplay(heightPrev);

  for (auto & node : indexNode) {
    node.Consume(preTree, samplePred, bottom);
  }
}


/**
  @brief Consumes all remaining information for node in the current level.

  @return void.
*/
void IndexNode::NonTerminal(PreTree *preTree, SamplePred *samplePred, Bottom *bottom) {
  if (ssNode != 0) {
    ssNode->NonTerminal(samplePred, preTree, bottom->Runs(), ptId, ptL, ptR);
  }
}


/**
  @brief Consumes all remaining information for node in the current level.

  @return void.
*/
void IndexNode::Consume(PreTree *preTree, SamplePred *samplePred, Bottom *bottom) {
  if (ssNode != 0) {
    lhSum = ssNode->Replay(samplePred, preTree, bottom->Runs(), idxCount, sum, ptId, ptL, ptR);
  }
}


void Index::LevelProduce(Bottom *bottom, unsigned int splitNext, unsigned int posRight, unsigned int idxLive, unsigned int idxMax) {
  bottom->Overlap(splitNext, idxLive, idxMax);

  unsigned int posLeft = 0;
  std::vector<IndexNode> indexNext(splitNext);
  for (auto & node : indexNode) {
    node.Produce(indexNext, posLeft, posRight);
  }
  indexNode = std::move(indexNext);

  bottom->PathUpdate(indexNode, preTree, levelWidth);
}


/**
   @brief Splitable nodes only:  takes census next level's left, right split
   nodes nodes and leaves.

   @param rhSplitNext outputs count of RH index nodes in next level.

   @param leafNext outputs count of pretree terminals in next level.

   @return void, plus output reference parameters.
*/
void IndexNode::Produce(std::vector<IndexNode> &indexNext, unsigned int &posLeft, unsigned int &posRight) const {
  if (ssNode != 0) {
    SplitHand(indexNext, posLeft, lhSCount, lhIdxCount, lhStart, ssNode->MinInfo(), ptL, lhSum, PathLeft());
    SplitHand(indexNext, posRight, sCount - lhSCount, idxCount - lhIdxCount, lhStart + lhIdxCount, ssNode->MinInfo(), ptR, sum - lhSum, PathRight());
  }
}


/**
   @brief Consults pretree for node holding sample and computes node's offset.

   @param sIdx is the sample index.

   @param levelOff is the level-relative offset of node holding sample index, if any.

   @return true iff node holding sample is at current level.
 */
bool Index::LevelOffSample(unsigned int sIdx, unsigned int &levelOffset) const {
  bool dummy;
  levelOffset = preTree->SampleOffset(sIdx, dummy);
  return levelOffset < levelWidth;
}


/**
   @brief Returns the level-relative offset associated with an index node.

   @param splitIdx is the split index referenced.

   @return pretree offset from level base.
  */

unsigned int Index::LevelOffSplit(unsigned int splitIdx) const {
  return preTree->LevelOffset(indexNode[splitIdx].PTId());
}
