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
#include "path.h"

#include <numeric>

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
IndexLevel::IndexLevel(int _nSamp, unsigned int _bagCount, double _sum) : indexSet(std::vector<IndexSet>(1)), bagCount(_bagCount), relBase(std::vector<unsigned int>(1)), rel2ST(std::vector<unsigned int>(bagCount)), st2Split(std::vector<unsigned int>(bagCount)) {
  indexSet[0].Init(0, _nSamp, 0, bagCount, 0.0, 0, _sum, 0, 0, bagCount);
  relBase[0] = 0;
  std::iota(rel2ST.begin(), rel2ST.end(), 0);
  std::fill(st2Split.begin(), st2Split.end(), 0);
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

    unsigned int leafNext;
    unsigned int splitNext = SplitCensus(argMax, leafNext);
    Consume(bottom, preTree, splitNext, leafNext);
    Produce(bottom, preTree, splitNext, level + 1 == totLevels);
  }
}


/**
   @brief Tallies previous level's splitting results.

   @param argMax is a vector of split signatures corresponding to the
   nodes.

   @return count of splitable nodes in the next level.
 */
unsigned int IndexLevel::SplitCensus(std::vector<SSNode*> &argMax, unsigned int &leafNext) {
  unsigned int splitNext, leafThis, idxExtent;
  leafThis = splitNext = idxExtent = idxLive = idxMax = 0;
  for (auto & iSet : indexSet) {
    iSet.SplitCensus(argMax, leafThis, splitNext, idxExtent, idxLive, idxMax);
  }

  // Restaging is implemented as a patient stable partition.
  //
  // Coprocessor implementations can be streamlined using an iSet-
  // independent indexing scheme, e.g., enumerating all left-hand
  // subnodes before the first right-hand subnode.
  //
  leafNext = 2 * (indexSet.size() - leafThis) - splitNext;

  succBase = std::move(std::vector<unsigned int>(splitNext + leafNext));
  std::fill(succBase.begin(), succBase.end(), idxExtent); // Inattainable base.

  succST = std::move(std::vector<unsigned int>(idxExtent));
  std::fill(succST.begin(), succST.end(), idxExtent); // Inattainable st index.

  return splitNext;
}


/**
   @brief Consumes relevant contents of split signature, if any, and accumulates
   leaf and splitting census.

   @param splitNext counts splitable nodes precipitated in the next level.

   @return void.
 */
void IndexSet::SplitCensus(std::vector<SSNode*> &argMax, unsigned int &leafThis, unsigned int &splitNext, unsigned int &idxExtent, unsigned int &idxLive, unsigned int &idxMax) {
  ssNode = argMax[splitIdx];
  if (ssNode == 0) {
    leafThis++;
  }
  else {
    idxExtent += extent;
    ssNode->LHSizes(lhSCount, lhExtent);
    splitNext += SplitAccum(lhExtent, idxLive, idxMax);
    splitNext += SplitAccum(extent - lhExtent, idxLive, idxMax);
  }
}


/**
   @brief Consumes current level of splits into new pretree level,
   then replays successor mappings.

   @param terminal is true iff no attempt will be made to split the
   new level's nodes.

   @return void.
*/
void IndexLevel::Consume(Bottom *bottom, PreTree *preTree, unsigned int splitNext, unsigned int leafNext) {
  bottom->Overlap(preTree, splitNext, leafNext); // Two levels co-exist.
  succLive = 0;
  succExtinct = splitNext; // Pseudo-indexing for extinct sets.
  liveBase = 0;
  extinctBase = idxLive;
  for (auto  & iSet : indexSet) {
    iSet.Consume(this, bottom, preTree);
  }

  bottom->Reindex(this);
  relBase = std::move(succBase);
  rel2ST = std::move(succST);
}


/**
  @brief Consumes iSet contents into pretree or terminal map.

  @return void.
*/
void IndexSet::Consume(IndexLevel *indexLevel, Bottom *bottom, PreTree *preTree) {
  if (ssNode != 0) {
    NonTerminal(indexLevel, bottom, preTree);
  }
  else {
    Terminal(indexLevel, bottom);
  }
}


/**
   @brief Caches state necessary for reindexing and useful subsequently.

   @return void.
 */
void IndexSet::NonTerminal(IndexLevel *indexLevel, Bottom *bottom, PreTree *preTree) {
  succLeft = indexLevel->IdxSucc(lhExtent);
  succRight = indexLevel->IdxSucc(extent - lhExtent);

  lhSum = bottom->NonTerminal(preTree, ssNode, extent, lhExtent, sum, ptId);
  leftExpl = ssNode->LeftExpl();
  offExpl = indexLevel->SuccBase(leftExpl ? succLeft : succRight);
  offImpl = indexLevel->SuccBase(leftExpl ? succRight : succLeft);
  pathLeft = IdxPath::PathNext(path, true);
  pathRight = IdxPath::PathNext(path, false);
}


/**
   @brief Builds index base offsets to mirror crescent pretree level.

   @return void.
 */
unsigned int IndexLevel::IdxSucc(unsigned int extent) {
  unsigned int idxSucc;
  if (IndexSet::Splitable(extent)) {
    idxSucc = succLive++;
    succBase[idxSucc] = liveBase;
    liveBase += extent;
  }
  else {
    idxSucc = succExtinct++;
    succBase[idxSucc] = extinctBase;
    extinctBase += extent;
  }

  return idxSucc;
}


/**
   @brief Dispatches index set to frontier.

   @return void.
 */
void IndexSet::Terminal(IndexLevel *indexLevel, Bottom *bottom) {
  indexLevel->Terminal(bottom, splitIdx, extent, ptId);
}


void IndexLevel::Terminal(Bottom *bottom, unsigned int splitIdx, unsigned int extent, unsigned int ptId) {
  unsigned int nodeBase = RelBase(splitIdx);
  for (unsigned int relIdx = nodeBase; relIdx < nodeBase + extent; relIdx++) {
    bottom->SetExtinct(relIdx, rel2ST[relIdx]);
  }
  bottom->Terminal(extent, ptId);
}


void IndexLevel::Reindex(Bottom *bottom, BV *replayExpl) {
  unsigned int i;
#pragma omp parallel default(shared) private(i)
  {
#pragma omp for schedule(dynamic, 1)
    for (i = 0; i < indexSet.size(); i++) {
      indexSet[i].Reindex(rel2ST, bottom, replayExpl, idxLive, succST);
    }
  }
}


/**
   @brief Subtree-relative reindexing:  indices randomly distributed
   among nodes (i.e., index sets).

   @param replayExpl sets high bits for those indices lying in the
   explicit portion of a split.

   @param stPath maps an index from the current level to its position
   in the upcoming level and to the recent reaching path.

   @return void.
 */
void IndexLevel::Reindex(BV *replayExpl, IdxPath *stPath) {
  for (unsigned int stIdx = 0; stIdx < bagCount; stIdx++) {
    unsigned int pathSucc, idxSucc;
    if (stPath->IsLive(stIdx)) {
      unsigned int splitIdx = st2Split[stIdx];
      st2Split[stIdx] = indexSet[splitIdx].Offspring(replayExpl->TestBit(stIdx), pathSucc, idxSucc);
      succST[idxSucc] = stIdx; // Write address staggered, but sequential.
      if (idxSucc < idxLive) { // Extinct nodes marked en masse by Terminal()
        stPath->SetLive(stIdx, pathSucc, idxSucc);
      }
    }
  }
}


/**
   @brief Node-relative reindexing:  indices contiguous on nodes (index sets).
 */
void IndexSet::Reindex(const std::vector<unsigned int> &rel2ST, Bottom *bottom, BV *replayExpl, unsigned int idxLive, std::vector<unsigned int> &succST) {
  if (ssNode == 0)
    return;

  Level *levelFront = bottom->LevelFront();
  unsigned int baseExpl = offExpl;
  unsigned int baseImpl = offImpl;
  for (unsigned int relIdx = relBase; relIdx < relBase + extent; relIdx++) {
    unsigned int stIdx = rel2ST[relIdx];
    bool expl = replayExpl->TestBit(relIdx);
    unsigned int targIdx = expl ? offExpl++ : offImpl++;
    succST[targIdx] = stIdx;

    // Live index updates could be deferred to production of the next level,
    // but the relIdx-to-targIdx mapping is conveniently available here.
    //
    // Extinct subtree indices are best updated on a per-node basis, however,
    // so their treatment is deferred until production of the next level.
    // Extinct node-relative indices, however, can be treated on the fly.
    //
    if (targIdx >= idxLive) {
      levelFront->SetExtinct(relIdx);
    }
    else {
      bottom->SetLive(relIdx, stIdx, (expl && leftExpl) || !(expl || leftExpl) ? pathLeft : pathRight, targIdx, expl ? baseExpl : baseImpl);
    }
  }
}


/**
   @brief Diagnostic test for replay.  Checks that left and right successors
   receive the expected index counts.

   @return count of mismatched expectations.

unsigned int Bottom::DiagReindex(const PreTree *preTree, unsigned int offExpl, unsigned int offImpl, bool leftExpl, unsigned int ptId, unsigned int lhExtent, unsigned int rhExtent) {
  unsigned int mismatch = 0;
  unsigned int extentImpl = leftExpl ? rhExtent : lhExtent;
  unsigned int extentExpl = leftExpl ? lhExtent : rhExtent;
  unsigned int ptImpl = leftExpl ? preTree->RHId(ptId) : preTree->LHId(ptId);
  unsigned int ptExpl = leftExpl ? preTree->LHId(ptId) : preTree->RHId(ptId);
  if (offExpl != SuccBase(ptExpl) + extentExpl) {
    mismatch++;
  }
  if (offImpl != SuccBase(ptImpl) + extentImpl) {
    mismatch++;
  }

  return mismatch;
}
*/

/**
   @brief Produces next level's index sets, as appropriate, and
   dispatches extinct nodes to pretree frontier.

   @return void.
 */
void IndexLevel::Produce(Bottom *bottom, PreTree *preTree, unsigned int splitNext, bool terminal) {
  bottom->LevelPrepare(terminal ? 0 : splitNext, idxLive, idxMax);
  std::vector<IndexSet> indexNext(bottom->SplitCount());
  for (auto & iSet : indexSet) {
    iSet.Produce(this, bottom, preTree, indexNext);
  }
  indexSet = std::move(indexNext);

  // Overlap persists through production of next level.
  //
  bottom->LevelClear();
}


/**
   @brief Produces next level's iSets for LH and RH sides of a split.

   @param indexNext is the crescent successor level of index sets.

   @return void, plus output reference parameters.
*/
void IndexSet::Produce(IndexLevel *indexLevel, Bottom *bottom, const PreTree *preTree, std::vector<IndexSet> &indexNext) const {
  if (ssNode != 0) {
    Successor(indexLevel, indexNext, succLeft, bottom, lhSCount, lhStart, lhExtent, ssNode->MinInfo(), preTree->LHId(ptId), lhSum, pathLeft);
    Successor(indexLevel, indexNext, succRight, bottom, sCount - lhSCount, lhStart + lhExtent, extent - lhExtent, ssNode->MinInfo(), preTree->RHId(ptId), sum - lhSum, pathRight);
  }
}


/**

   @brief Appends one hand of a split onto next level's iSet list, if
   splitable, otherwise dispatches a terminal iSet.

   @return void.
*/
void IndexSet::Successor(IndexLevel *indexLevel, std::vector<IndexSet> &indexNext, unsigned int succIdx, Bottom *bottom, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) const {
  if (succIdx >= indexNext.size()) { // Extinct.
    indexLevel->Terminal(bottom, succIdx, _extent, _ptId);
  }
  else {
    indexNext[succIdx].SuccInit(indexLevel, bottom, succIdx, splitIdx, _sCount, _lhStart, _extent, _minInfo, _ptId, _sum, _path);
  }
}


/**
   @brief Initializes index set as a successor node.

   @return void.
 */
void IndexSet::SuccInit(IndexLevel *indexLevel, Bottom *bottom, unsigned int _splitIdx, unsigned int parIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) {
  Init(_splitIdx, _sCount, _lhStart, _extent, _minInfo, _ptId, _sum, _path, indexLevel->RelBase(_splitIdx), indexLevel->BagCount());
  bottom->ReachingPath(splitIdx, parIdx, lhStart, extent, relBase, path);
}
