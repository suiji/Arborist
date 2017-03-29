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
unsigned int IndexLevel::minNode = 0;

/**
   @brief Initialization of static invariants.

   @param _minNode is the minimum node size for splitting.

   @param _totLevels is the maximum number of levels to evaluate.

   @return void.
 */
void IndexLevel::Immutables(unsigned int _minNode, unsigned int _totLevels) {
  minNode = _minNode;
  totLevels = _totLevels;
}


/**
   @brief Reset of statics.

   @return void.
 */
void IndexLevel::DeImmutables() {
  totLevels = 0;
  minNode = 0;
}



/**
   @brief Per-tree constructor.  Sets up root node for level zero.
 */
IndexLevel::IndexLevel(const std::vector<SampleNode> &_stageSample, unsigned int nSamp, double bagSum) : stageSample(_stageSample), indexSet(std::vector<IndexSet>(1)), bagCount(stageSample.size()), idxLive(bagCount), relBase(std::vector<unsigned int>(1)), rel2ST(std::vector<unsigned int>(bagCount)), rel2Sample(stageSample), st2Split(std::vector<unsigned int>(bagCount)) {
  indexSet[0].Init(0, nSamp, 0, bagCount, 0.0, 0, bagSum, 0, 0, bagCount);
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
    ptBlock[blockIdx] = OneTree(pmTrain, sample);
  }
  
  return ptBlock;
}


/**
   @brief Performs sampling and level processing for a single tree.

   @return void.
 */
PreTree *IndexLevel::OneTree(const PMTrain *pmTrain, Sample *sample) {
  PreTree *preTree = new PreTree(pmTrain, sample->BagCount());
  IndexLevel *index = new IndexLevel(sample->StageSample(), Sample::NSamp(), sample->BagSum());
  Bottom *bottom = sample->Bot();
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
    //    cout << "\nLevel " << level << "\n" << endl;
    std::vector<SSNode*> argMax(indexSet.size());
    bottom->Split(*this, argMax);

    unsigned int leafNext;
    unsigned int splitNext = SplitCensus(argMax, leafNext, level + 1 == totLevels);
    Consume(bottom, preTree, splitNext, leafNext);
    Produce(bottom, preTree, splitNext);
  }
}


/**
   @brief Tallies previous level's splitting results.

   @param argMax is a vector of split signatures corresponding to the
   nodes.

   @return count of splitable nodes in the next level.
 */
unsigned int IndexLevel::SplitCensus(std::vector<SSNode*> &argMax, unsigned int &leafNext, bool _levelTerminal) {
  levelTerminal = _levelTerminal;
  unsigned int splitNext, leafThis, idxExtent;
  idxExtent = idxLive; // Previous level's index space.
  leafThis = splitNext = idxLive = idxMax = 0;
  for (auto & iSet : indexSet) {
    iSet.SplitCensus(argMax, this, leafThis, splitNext, idxLive, idxMax);
  }

  // Restaging is implemented as a patient stable partition.
  //
  // Coprocessor implementations can be streamlined using an iSet-
  // independent indexing scheme, e.g., enumerating all left-hand
  // subnodes before the first right-hand subnode.
  //
  leafNext = 2 * (indexSet.size() - leafThis) - splitNext;

  succBase = std::move(std::vector<unsigned int>(splitNext + leafNext + leafThis));
  std::fill(succBase.begin(), succBase.end(), idxExtent); // Inattainable base.

  return splitNext;
}


/**
   @brief Consumes relevant contents of split signature, if any, and accumulates
   leaf and splitting census.

   @param splitNext counts splitable nodes precipitated in the next level.

   @return void.
 */
void IndexSet::SplitCensus(std::vector<SSNode*> &argMax, IndexLevel *indexLevel, unsigned int &leafThis, unsigned int &splitNext, unsigned int &idxLive, unsigned int &idxMax) {
  ssNode = argMax[splitIdx];
  if (ssNode == 0) {
    leafThis++;
  }
  else {
    ssNode->LHSizes(lhSCount, lhExtent);
    splitNext += SplitAccum(indexLevel, lhExtent, idxLive, idxMax);
    splitNext += SplitAccum(indexLevel, extent - lhExtent, idxLive, idxMax);
  }
}


  /**
     @return count of splitable nodes precipitated in next level:  0/1.
   */
unsigned IndexSet::SplitAccum(class IndexLevel *indexLevel, unsigned int _extent, unsigned int &_idxLive, unsigned int &_idxMax) {
    if (indexLevel->Splitable(_extent)) {
      _idxLive += _extent;
      _idxMax = _extent > _idxMax ? _extent : _idxMax;
      return 1;
    }
    else {
      return 0;
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
  leftExpl = bottom->NonTerminal(preTree, ssNode, extent, ptId, sumExpl);

  succExpl = indexLevel->IdxSucc(bottom, leftExpl ? lhExtent : extent - lhExtent, leftExpl ? preTree->LHId(ptId) : preTree->RHId(ptId), offExpl);
  succImpl = indexLevel->IdxSucc(bottom, leftExpl ? extent - lhExtent : lhExtent, leftExpl ? preTree->RHId(ptId) : preTree->LHId(ptId), offImpl);

  pathExpl = IdxPath::PathNext(path, leftExpl);
  pathImpl = IdxPath::PathNext(path, !leftExpl);
}


/**
   @brief Builds index base offsets to mirror crescent pretree level.

   @param extent is the count of the index range.

   @param ptId is the index of the corresponding pretree node.

   @param offOut outputs the node-relative starting index.  Should not
   exceed 'idxExtent', the live high watermark of the previous level.

   @return void.
 */
unsigned int IndexLevel::IdxSucc(Bottom *bottom, unsigned int extent, unsigned int ptId, unsigned int &offOut, bool terminal) {
  terminal |= !Splitable(extent);
  unsigned int idxSucc;
  if (terminal) { // Pseudo split holds terminal settings.
    idxSucc = succExtinct++;
    offOut = extinctBase;
    extinctBase += extent;
    bottom->Terminal(offOut, extent, ptId); 
  }
  else {
    idxSucc = succLive++;
    offOut = liveBase;
    liveBase += extent;
  }
  succBase[idxSucc] = offOut;
  
  return idxSucc;
}


/**
   @brief Dispatches index set to frontier.

   @return void.
 */
void IndexSet::Terminal(IndexLevel *indexLevel, Bottom *bottom) {
  succOnly = indexLevel->IdxSucc(bottom, extent, ptId, offOnly, true);
}


/**
   @brief Driver for node-relative reindexing.
 */
void IndexLevel::Reindex(Bottom *bottom, BV *replayExpl) {
  std::vector<unsigned int> succST(idxLive);
  std::vector<SampleNode> succSample(idxLive);

  unsigned int i;
#pragma omp parallel default(shared) private(i)
  {
#pragma omp for schedule(dynamic, 1)
    for (i = 0; i < indexSet.size(); i++) {
      indexSet[i].Reindex(bottom, replayExpl, idxLive, rel2ST, succST, rel2Sample, succSample);
    }
  }
  rel2ST = std::move(succST);
  rel2Sample = std::move(succSample);
}


/**
   @brief Node-relative reindexing:  indices contiguous on nodes (index sets).
 */
void IndexSet::Reindex(Bottom *bottom, BV *replayExpl, unsigned int idxLive, const std::vector<unsigned int> &rel2ST, std::vector<unsigned int> &succST, const std::vector<SampleNode> &rel2Sample, std::vector<SampleNode> &succSample) {
  if (ssNode == 0) {
    TerminalReindex(bottom, rel2ST);
  }
  else {
    NonterminalReindex(bottom, replayExpl, idxLive, rel2ST, succST, rel2Sample, succSample);
  }
}


void IndexSet::TerminalReindex(Bottom *bottom, const std::vector<unsigned int> &rel2ST) {
  for (unsigned int relIdx = relBase; relIdx < relBase + extent; relIdx++) {
    bottom->SetExtinct(relIdx, offOnly++, rel2ST[relIdx]);
  }
}


void IndexSet::NonterminalReindex(Bottom *bottom, BV *replayExpl, unsigned int idxLive, const std::vector<unsigned int> &rel2ST, std::vector<unsigned int> &succST, const std::vector<SampleNode> &rel2Sample, std::vector<SampleNode> &succSample) {
  unsigned int baseExpl = offExpl;
  unsigned int baseImpl = offImpl;
  for (unsigned int relIdx = relBase; relIdx < relBase + extent; relIdx++) {
    unsigned int stIdx = rel2ST[relIdx];
    bool expl = replayExpl->TestBit(relIdx);
    unsigned int targIdx = expl ? offExpl++ : offImpl++;

    if (targIdx < idxLive) {
      succST[targIdx] = stIdx;
      succSample[targIdx] = rel2Sample[relIdx]; 
      bottom->SetLive(relIdx, targIdx, stIdx, expl ? pathExpl : pathImpl, expl ? baseExpl : baseImpl);
    }
    else {
      bottom->SetExtinct(relIdx, targIdx, stIdx);
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
void IndexLevel::Reindex(Bottom *bottom, BV *replayExpl, IdxPath *stPath) {
  for (unsigned int stIdx = 0; stIdx < bagCount; stIdx++) {
    unsigned int pathSucc, idxSucc;
    if (stPath->IsLive(stIdx)) {
      unsigned int splitIdx = st2Split[stIdx];
      st2Split[stIdx] = indexSet[splitIdx].Offspring(replayExpl->TestBit(stIdx), pathSucc, idxSucc);
      if (idxSucc < idxLive) {
	rel2ST[idxSucc] = stIdx; // Needed for transition.
	rel2Sample[idxSucc] = stageSample[stIdx]; // semi-regular target access.
        stPath->SetLive(stIdx, pathSucc, idxSucc);
      }
      else {
	bottom->SetExtinct(idxSucc, stIdx);
      }
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
void IndexLevel::Produce(Bottom *bottom, PreTree *preTree, unsigned int splitNext) {
  bottom->LevelPrepare(splitNext, idxLive, idxMax);
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
    Successor(indexLevel, indexNext, leftExpl ? succExpl : succImpl, bottom, lhSCount, lhStart, lhExtent, ssNode->MinInfo(), preTree->LHId(ptId), leftExpl ? sumExpl : sum - sumExpl, leftExpl ? pathExpl : pathImpl);
    Successor(indexLevel, indexNext, leftExpl ? succImpl : succExpl, bottom, sCount - lhSCount, lhStart + lhExtent, extent - lhExtent, ssNode->MinInfo(), preTree->RHId(ptId), leftExpl ? sum - sumExpl : sumExpl, leftExpl ? pathImpl : pathExpl);
  }
}


/**

   @brief Appends one hand of a split onto next level's iSet list, if
   splitable, otherwise dispatches a terminal iSet.

   @return void.
*/
void IndexSet::Successor(IndexLevel *indexLevel, std::vector<IndexSet> &indexNext, unsigned int succIdx, Bottom *bottom, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path) const {
  if (succIdx < indexNext.size()) {
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


/**
   @brief Visits all live indices, so likely worth parallelizing.
 */
void IndexLevel::SumsAndSquares(unsigned int ctgWidth, std::vector<double> &sumSquares, std::vector<double> &ctgSum, std::vector<bool> &unsplitable) const {
  unsigned int splitIdx;
  
#pragma omp parallel default(shared) private(splitIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
    unsplitable[splitIdx] = indexSet[splitIdx].SumsAndSquares(rel2Sample, ctgWidth, sumSquares[splitIdx], &ctgSum[splitIdx * ctgWidth]);
    }
  }
}


/**
   @brief Sums each category for a node splitable in the upcoming level.

   @param sumSquares accumulates the sum of squares over each category.
   Assumed intialized to zero.

   @param ctgSum records the response sums, by category.  Assumed initialized
   to zero.

   @return true iff response constrained to a single category.
   
*/
bool IndexSet::SumsAndSquares(const std::vector<SampleNode> &rel2Sample, unsigned int ctgWidth, double  &sumSquares, double *ctgSum) const {
  std::vector<unsigned int> sCountCtg(ctgWidth);
  std::fill(sCountCtg.begin(), sCountCtg.end(), 0);

  for (unsigned int relIdx = 0; relIdx < extent; relIdx++) {
    FltVal idxSum;
    unsigned int idxSCount;
    unsigned int ctg = rel2Sample[relBase + relIdx].Ref(idxSum, idxSCount);
    ctgSum[ctg] += idxSum;
    sCountCtg[ctg] += idxSCount;
  }

  bool unsplitable = false;
  for (unsigned int ctg = 0; ctg < ctgWidth; ctg++) {
    if (sCountCtg[ctg] == sCount) {
      unsplitable = true; // Short-circuits singleton response.
    }
    sumSquares += ctgSum[ctg] * ctgSum[ctg];
  }

  return unsplitable;
}

