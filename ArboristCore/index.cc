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
#include "bottom.h"

// Testing only:
//#include <iostream>
//using namespace std;
//#include <time.h>
//clock_t clock(void);


unsigned int IndexLevel::totLevels = 0;
unsigned int IndexSet::minNode = 0;

/**
   @brief Initialization of static invariants.

   @param _minNode is the minimum node size for splitting.

   @param _totLevels is the maximum number of levels to evaluate.

   @return void.
 */
void IndexLevel::Immutables(unsigned int _minNode, unsigned int _totLevels) {
  IndexSet::minNode = _minNode;
  totLevels = _totLevels;
}


/**
   @brief Reset of statics.

   @return void.
 */
void IndexLevel::DeImmutables() {
  totLevels = 0;
  IndexSet::minNode = 0;
}



/**
   @brief Per-tree constructor.  Sets up root node for level zero.
 */
IndexLevel::IndexLevel(int _nSamp, unsigned int _bagCount, double _sum) : indexSet(std::vector<IndexSet>(1)), bagCount(_bagCount) {
  indexSet[0].Init(0, _nSamp, 0, bagCount, 0.0, 0, _sum, 0);
}


/**
   @brief Destructor.

   @return void.
 */
IndexLevel::~IndexLevel() {
}


IndexSet::IndexSet() : preBias(0.0), splitIdx(0), ptId(0), lhStart(0), extent(0), sCount(0), sum(0.0), minInfo(0.0), path(0), ssNode(0) {
}


/**
   @brief Instantiates a block of PreTees for bulk return, but may or may
   not build them concurrently.

   @param sampleBlock contains the sample objects characterizing the roots.

   @param treeBlock is the number of trees to train in this block.

   @return brace of 'treeBlock'-many PreTree objects.
*/
PreTree **IndexLevel::BlockTrees(const PMTrain *pmTrain, Sample **sampleBlock, int treeBlock) {
  PreTree **ptBlock = new PreTree*[treeBlock];

  for (int blockIdx = 0; blockIdx < treeBlock; blockIdx++) {
    Sample *sample = sampleBlock[blockIdx];
    ptBlock[blockIdx] = OneTree(pmTrain, sample->Bot(), Sample::NSamp(), sample->BagCount(), sample->BagSum());
  }
  
  return ptBlock;
}


/**
   @brief Performs sampling and level processing for a single tree.

   @return void.
 */
PreTree *IndexLevel::OneTree(const PMTrain *pmTrain, Bottom *bottom, int _nSamp, unsigned int _bagCount, double _sum) {
  PreTree *preTree = new PreTree(pmTrain, _bagCount);
  IndexLevel *index = new IndexLevel(_nSamp, _bagCount, _sum);
  index->Levels(bottom, preTree);
  delete index;

  bottom->SubtreeFrontier(preTree);

  return preTree;
}


/**
   @brief Main loop for per-level splitting.  Assumes root node and
   attendant per-tree data structures have been initialized.

   @return void.
*/
void  IndexLevel::Levels(Bottom *bottom, PreTree *preTree) {
  for (unsigned int level = 0; !indexSet.empty(); level++) {
    //cout << "\nLevel " << level << "\n" << endl;
    std::vector<SSNode*> argMax(indexSet.size());
    bottom->Split(*this, argMax);

    unsigned int leafNext, idxExtent, idxLive;
    unsigned int splitNext = SplitCensus(argMax, leafNext, idxExtent, idxLive);
    Consume(bottom, preTree, splitNext, leafNext, idxExtent, idxLive, level + 1 == totLevels);

    Produce(bottom, preTree);
  }
}


/**
   @brief Tallies previous level's splitting results.

   @param argMax is a vector of split signatures corresponding to the
   nodes.

   @return count of splitable nodes in the next level.
 */
unsigned int IndexLevel::SplitCensus(const std::vector<SSNode*> &argMax, unsigned int &leafNext, unsigned int &idxExtent, unsigned int &idxLive) {
  unsigned int lhSplitNext, rhSplitNext, leafThis, splitIdx;
  rhSplitNext = leafThis = lhSplitNext = idxExtent = idxLive = splitIdx = 0;
  for (auto & iSet : indexSet) {
    iSet.SplitCensus(argMax[splitIdx++], leafThis, lhSplitNext, rhSplitNext, idxExtent, idxLive);
  }

  // Restaging is implemented as a patient stable partition.
  //
  // Coprocessor implementations can be streamlined using an iSet-
  // independent indexing scheme, e.g., enumerating all left-hand
  // subnodes before the first right-hand subnode.
  //
  unsigned int splitNext = lhSplitNext + rhSplitNext;
  leafNext = 2 * (indexSet.size() - leafThis) - splitNext;

  return splitNext;
}


/**
   @brief Consumes relevant contents of split signature, if any, and accumulates
   leaf and splitting census.

   @param lhSplitNext counts splitable LH nodes precipitated in the next leve.

   @param rhSplitNext counts RH.
 */
void IndexSet::SplitCensus(SSNode *argMax, unsigned int &leafThis, unsigned int &lhSplitNext, unsigned int &rhSplitNext, unsigned int &idxExtent, unsigned int &idxLive) {
  ssNode = argMax;
  if (ssNode == 0) {
    leafThis++;
  }
  else {
    idxExtent += extent;
    ssNode->LHSizes(lhSCount, lhExtent);
    lhSplitNext += SplitAccum(lhExtent, idxLive);
    rhSplitNext += SplitAccum(extent - lhExtent, idxLive);
  }
}


/**
   @brief Consumes current level of splits into new pretree level,
   then replays successor mappings.

   @param terminal is true iff no attempt will be made to split the
   new level's nodes.

   @return void.
*/
void IndexLevel::Consume(Bottom *bottom, PreTree *preTree, unsigned int splitNext, unsigned int leafNext, unsigned int idxExtent, unsigned int idxLive, bool terminal) {
  bottom->LevelSucc(preTree, splitNext, leafNext, idxExtent, idxLive, terminal);
  for (auto  & iSet : indexSet) {
    iSet.Consume(bottom, preTree);
  }

  unsigned int i;
#pragma omp parallel default(shared) private(i)
  {
#pragma omp for schedule(dynamic, 1)
    for (i = 0; i < indexSet.size(); i++) {
      indexSet[i].Replay(bottom, preTree);
    }
  }
  
  bottom->Overlap(terminal ? 0 : splitNext);
}


/**
  @brief Consumes iSet contents into pretree or terminal map.

  @return void.
*/
void IndexSet::Consume(Bottom *bottom, PreTree *preTree) {
  if (ssNode != 0) {
    NonTerminal(bottom, preTree);
  }
  else {
    Terminal(bottom);
  }
}


/**
 */
void IndexSet::NonTerminal(Bottom *bottom, PreTree *preTree) {
  lhSum = bottom->NonTerminal(preTree, ssNode, extent, lhExtent, sum, ptId);
}


/**
   @brief Dispatches index set to frontier.

   @return void.
 */
void IndexSet::Terminal(Bottom *bottom) {
  bottom->Terminal(extent, ptId);
}


void IndexSet::Replay(Bottom *bottom, const PreTree *preTree) {
  if (ssNode != 0) {
    bottom->Replay(preTree, ptId, path, ssNode->LeftExpl(), extent, lhExtent);
  }
}


/**
 */
void IndexLevel::Produce(Bottom *bottom, PreTree *preTree) {
  std::vector<IndexSet> indexNext;
  for (auto & iSet : indexSet) {
    iSet.Produce(bottom, preTree, indexNext);
  }
  indexSet = std::move(indexNext);

  bottom->LevelClear();
}


/**
   @brief Produces next level's iSets for LH and RH sides of a split.

   @param indexNext is the crescent successor level of index sets.

   @return void, plus output reference parameters.
*/
void IndexSet::Produce(Bottom *bottom, const PreTree *preTree, std::vector<IndexSet> &indexNext) const {
  if (ssNode != 0) {
    Successor(indexNext, bottom, lhSCount, lhStart, lhExtent, ssNode->MinInfo(), preTree->LHId(ptId), lhSum, PathLeft());
    Successor(indexNext, bottom, sCount - lhSCount, lhStart + lhExtent, extent - lhExtent, ssNode->MinInfo(), preTree->RHId(ptId), sum - lhSum, PathRight());
  }
}


/**

   @brief Appends one hand of a split onto next level's iSet list, if
   splitable, otherwise dispatches a terminal iSet.

   @return void.
*/
void IndexSet::Successor(std::vector<IndexSet> &indexNext, Bottom *bottom, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) const {
  if (!bottom->IsLive(_ptId)) {
    bottom->Terminal(_extent, _ptId);
  }
  else { // TODO:  Why must successor's internal index match vector position?
    IndexSet succ;
    succ.SuccInit(bottom, indexNext.size(), splitIdx, _sCount, _lhStart, _extent, _minInfo, _ptId, _sum, _path);
    indexNext.push_back(succ);
  }
}


/**
   @brief Initializes index set as a successor node.

   @return void.
 */
void IndexSet::SuccInit(Bottom *bottom, unsigned int _splitIdx, unsigned int parIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) {
  Init(_splitIdx, _sCount, _lhStart, _extent, _minInfo, _ptId, _sum, _path);
  bottom->ReachingPath(splitIdx, parIdx, lhStart, extent, ptId, path);
}


/**
   @brief Looks up subtree-relative index from node-relative coordinates.

   @param splitIdx is an IndexSet index.

   @param relIdx is an offset relative to the set index.

   @return subtree-relative index.
 */
unsigned int IndexLevel::STIdx(Bottom *bottom, unsigned int splitIdx, unsigned int relIdx) const {
  return bottom->STIdx(indexSet[splitIdx].PTId(), relIdx);
}
