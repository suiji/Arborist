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
#include "samplepred.h"
#include "splitsig.h"
#include "bottom.h"
#include "path.h"

#include <numeric>

// Testing only:
//#include <iostream>
//using namespace std;
//#include <time.h>


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
IndexLevel::IndexLevel(SamplePred *_samplePred, const std::vector<SumCount> &ctgRoot, Bottom *_bottom, unsigned int nSamp, unsigned int _bagCount, double bagSum) : samplePred(_samplePred), bottom(_bottom), indexSet(std::vector<IndexSet>(1)), bagCount(_bagCount), nodeRel(false), idxLive(bagCount), relBase(std::vector<unsigned int>(1)), rel2ST(std::vector<unsigned int>(bagCount)), st2Split(std::vector<unsigned int>(bagCount)), st2PT(std::vector<unsigned int>(bagCount)), replayExpl(new BV(bagCount))  {
  indexSet[0].Init(0, nSamp, 0, bagCount, 0.0, 0, bagSum, 0, 0, bagCount, ctgRoot, ctgRoot, true);
  relBase[0] = 0;
  std::iota(rel2ST.begin(), rel2ST.end(), 0);
  std::fill(st2Split.begin(), st2Split.end(), 0);
  std::fill(st2PT.begin(), st2PT.end(), 0);
}


/**
  @brief Sets fields with values used immediately following splitting.

  @return void.
 */
void IndexSet::Init(unsigned int _splitIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path, unsigned int _relBase, unsigned int bagCount, const std::vector<SumCount> &_ctgSum, const std::vector<SumCount> &_ctgExpl, bool explHand) {
    splitIdx = _splitIdx;
    sCount = _sCount;
    lhStart = _lhStart;
    extent = _extent;
    minInfo = _minInfo;
    ptId = _ptId;
    sum = _sum;
    path = _path;
    relBase = _relBase;
    if (explHand) {
      ctgSum = _ctgExpl;
    }
    else {
      ctgSum = _ctgSum;
      Decr(ctgSum, _ctgExpl);
    }
    ctgExpl = std::move(std::vector<SumCount>(ctgSum.size()));

    // Inattainable value.  Reset only when non-terminal:
    succExpl = succImpl = offExpl = offImpl = bagCount;
}

void IndexSet::Decr(std::vector<SumCount> &_ctgSum, const std::vector<SumCount> &_ctgSub) {
  unsigned i = 0;
  for (auto & sc : _ctgSum) {
    sc.Decr(_ctgSub[i++]);
  }
}


/**
   @brief Destructor.

   @return void.
 */
IndexLevel::~IndexLevel() {
  delete samplePred;
  delete bottom;
  delete replayExpl;
}


IndexSet::IndexSet() : preBias(0.0), splitIdx(0), ptId(0), lhStart(0), extent(0), sCount(0), sum(0.0), minInfo(0.0), path(0), unsplitable(false), sumExpl(0.0){
}


/**
   @brief Instantiates a block of PreTees for bulk return, but may or may
   not build them concurrently.

   @param sampleBlock contains the sample objects characterizing the roots.

   @param ptBlock is a brace of 'treeBlock'-many PreTree objects.
*/
void IndexLevel::TreeBlock(const PMTrain *pmTrain, const RowRank *rowRank, const std::vector<Sample *> &sampleBlock, const Coproc *coproc, std::vector<PreTree*> &ptBlock) {
  unsigned int blockIdx = 0;
  for (auto & sample : sampleBlock) {
    ptBlock[blockIdx++] = OneTree(pmTrain, rowRank, sample, coproc);
  }
}


/**
   @brief Performs sampling and level processing for a single tree.

   @return void.
 */
PreTree *IndexLevel::OneTree(const PMTrain *pmTrain, const RowRank *rowRank, const Sample *sample, const Coproc *coproc) {
  PreTree *preTree = new PreTree(pmTrain, sample->BagCount());
  IndexLevel *index = sample->IndexFactory(pmTrain, rowRank, coproc);
  index->Levels(rowRank, sample, preTree);
  delete index;
  
  return preTree;
}


/**
   @brief Main loop for per-level splitting.  Assumes root node and
   attendant per-tree data structures have been initialized.

   @return void.
*/
void  IndexLevel::Levels(const RowRank *rowRank, const Sample *sample, PreTree *preTree) {
  sample->Stage(rowRank, samplePred, bottom);
  for (unsigned int level = 0; !indexSet.empty(); level++) {
    //cout << "\nLevel " << level << "\n" << endl;
    bottom->LevelInit(this);
    std::vector<SSNode> argMax(indexSet.size());
    InfoInit(argMax);
    bottom->Split(samplePred, this, argMax);

    unsigned int leafNext, idxMax;
    unsigned int splitNext = SplitCensus(argMax, leafNext, idxMax, level + 1 == totLevels);
    Consume(preTree, argMax, splitNext, leafNext, idxMax);
    Produce(preTree, splitNext);
    bottom->LevelClear();
  }

  RelFlush();
  preTree->SubtreeFrontier(st2PT);
}


/**
   @brief Initializes splitting threshold on each of the arg-max nodes
   from associated splitting candidates.

   @return void.
 */
void IndexLevel::InfoInit(std::vector<SSNode> &argMax) const {
  for (auto & iSet : indexSet) {
    argMax[iSet.SplitIdx()].SetInfo(iSet.MinInfo());
  }
}


/**
   @brief Tallies previous level's splitting results.

   @param argMax is a vector of split signatures corresponding to the
   nodes.

   @return count of splitable nodes in the next level.
 */
unsigned int IndexLevel::SplitCensus(const std::vector<SSNode> &argMax, unsigned int &leafNext, unsigned int &idxMax, bool _levelTerminal) {
  levelTerminal = _levelTerminal;
  unsigned int splitNext, leafThis, idxExtent;
  idxExtent = idxLive; // Previous level's index space.
  leafThis = splitNext = idxLive = idxMax = 0;
  for (auto & iSet : indexSet) {
    iSet.ApplySplit(argMax);
    iSet.SplitCensus(this, leafThis, splitNext, idxLive, idxMax);
  }

  // Restaging is implemented as a patient stable partition.
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
void IndexSet::SplitCensus(IndexLevel *indexLevel, unsigned int &leafThis, unsigned int &splitNext, unsigned int &idxLive, unsigned int &idxMax) {
  if (!terminal) {
    splitNext += SplitAccum(indexLevel, lhExtent, idxLive, idxMax);
    splitNext += SplitAccum(indexLevel, extent - lhExtent, idxLive, idxMax);
  }
  else {
    leafThis++;
  }
}


/**
     @brief Sets members according to whether the set splits.

     @param argMax is the arg-max node for the split.

     @return void.
 */
void IndexSet::ApplySplit(const std::vector<SSNode> &argMaxVec) {
  SSNode argMax = argMaxVec[splitIdx];
  if (argMax.Info() > minInfo) {
    argMax.LHSizes(lhSCount, lhExtent);
    minInfo = argMax.MinInfo(); // Reset for splitting next level.
    terminal = false;
  }
  else {
    terminal = true;
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

   @return void.
*/
void IndexLevel::Consume(PreTree *preTree, const std::vector<SSNode> &argMax, unsigned int splitNext, unsigned int leafNext, unsigned int idxMax) {
  preTree->Level(splitNext, leafNext); // Overlap:  two levels co-exist.
  replayExpl->Clear();
  succLive = 0;
  succExtinct = splitNext; // Pseudo-indexing for extinct sets.
  liveBase = 0;
  extinctBase = idxLive;
  for (auto & iSet : indexSet) {
    iSet.Consume(this, bottom, preTree, argMax);
  }

  if (nodeRel) {
    NodeReindex();
  }
  else {
    nodeRel = IdxPath::Localizes(bagCount, idxMax);
    if (nodeRel) {
      TransitionReindex(splitNext);
    }
    else {
      SubtreeReindex(splitNext);
    }
  }

  relBase = std::move(succBase);
}


/**
  @brief Consumes iSet contents into pretree or terminal map.

  @return void.
*/
void IndexSet::Consume(IndexLevel *indexLevel, Bottom *bottom, PreTree *preTree, const std::vector<SSNode> &argMax) {
  if (!terminal) {
    NonTerminal(indexLevel, preTree, argMax[splitIdx]);
  }
  else {
    Terminal(indexLevel);
  }
}


/**
   @brief Dispatches index set to frontier.

   @return void.
 */
void IndexSet::Terminal(IndexLevel *indexLevel) {
  succOnly = indexLevel->IdxSucc(extent, ptId, offOnly, true);
}


/**
   @brief Caches state necessary for reindexing and useful subsequently.

   @return void.
 */
void IndexSet::NonTerminal(IndexLevel *indexLevel, PreTree *preTree, const SSNode &argMax) {
  leftExpl = indexLevel->NonTerminal(preTree, this, argMax);
  ptExpl = leftExpl ? preTree->LHId(ptId) : preTree->RHId(ptId);
  ptImpl = leftExpl ? preTree->RHId(ptId) : preTree->LHId(ptId);
  succExpl = indexLevel->IdxSucc(leftExpl ? lhExtent : extent - lhExtent, ptExpl, offExpl);
  succImpl = indexLevel->IdxSucc(leftExpl ? extent - lhExtent : lhExtent, ptImpl, offImpl);

  pathExpl = IdxPath::PathNext(path, leftExpl);
  pathImpl = IdxPath::PathNext(path, !leftExpl);
}


/**
   @param sumExpl outputs response sum over explicit hand of the split.

   @return true iff left hand of the split is explicit.
 */
bool IndexLevel::NonTerminal(PreTree *preTree, IndexSet *iSet, const SSNode &argMax) {
  return argMax.NonTerminal(this, preTree, iSet, bottom->Runs());
}


/**
   @brief Builds index base offsets to mirror crescent pretree level.

   @param extent is the count of the index range.

   @param ptId is the index of the corresponding pretree node.

   @param offOut outputs the node-relative starting index.  Should not
   exceed 'idxExtent', the live high watermark of the previous level.

   @param predTerminal is true iff predecessor node is terminal.

   @return void.
 */
unsigned int IndexLevel::IdxSucc(unsigned int extent, unsigned int ptId, unsigned int &offOut, bool predTerminal) {
  unsigned int idxSucc;
  if (predTerminal || !Splitable(extent)) { // Pseudo split caches settings.
    idxSucc = succExtinct++;
    offOut = extinctBase;
    extinctBase += extent;
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
   @brief Driver for node-relative reindexing.
 */
void IndexLevel::NodeReindex() {
  std::vector<unsigned int> succST(idxLive);
  rel2PT = std::move(std::vector<unsigned int>(idxLive));

  unsigned int splitIdx;
#pragma omp parallel default(shared) private(splitIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      indexSet[splitIdx].Reindex(replayExpl, this, idxLive, succST);
    }
  }
  rel2ST = std::move(succST);
}


/**
   @brief Node-relative reindexing:  indices contiguous on nodes (index sets).
 */
void IndexSet::Reindex(const BV *replayExpl, IndexLevel *index, unsigned int idxLive, std::vector<unsigned int> &succST) {
  if (terminal) {
    index->RelExtinct(relBase, extent, ptId);
  }
  else {
    NontermReindex(replayExpl, index, idxLive, succST);
  }
}


void IndexSet::NontermReindex(const BV *replayExpl, IndexLevel *index, unsigned int idxLive, std::vector<unsigned int> &succST) {
  unsigned int baseExpl = offExpl;
  unsigned int baseImpl = offImpl;
  for (unsigned int relIdx = relBase; relIdx < relBase + extent; relIdx++) {
    bool expl = replayExpl->TestBit(relIdx);
    unsigned int targIdx = expl ? offExpl++ : offImpl++;

    if (targIdx < idxLive) {
      succST[targIdx] = index->RelLive(relIdx, targIdx, expl ? pathExpl : pathImpl, expl? baseExpl : baseImpl, expl ? ptExpl : ptImpl);
    }
    else {
      index->RelExtinct(relIdx, expl ? ptExpl : ptImpl);
    }
  }
}


/**
   @brief Updates the mapping from live relative indices to associated
   PreTree indices.

   @return corresponding subtree-relative index.
*/
unsigned int IndexLevel::RelLive(unsigned int relIdx, unsigned int targIdx, unsigned int path, unsigned int base, unsigned int ptIdx) {
  unsigned int stIdx = rel2ST[relIdx];
  rel2PT[targIdx] = ptIdx;
  bottom->SetLive(relIdx, targIdx, stIdx, path, base);

  return stIdx;
}


/**
   @brief Translates node-relative back to subtree-relative indices on 
   terminatinal node.

   @param relIdx is the node-relative index.

   @param ptId is the pre-tree index of the associated node.

   @return void.
 */
void IndexLevel::RelExtinct(unsigned int relIdx, unsigned int ptId) {
  unsigned int stIdx = rel2ST[relIdx];
  st2PT[stIdx] = ptId;
  bottom->SetExtinct(relIdx, stIdx);
}


/**
   @brief Subtree-relative reindexing:  indices randomly distributed
   among nodes (i.e., index sets).

   @return void.
*/
void IndexLevel::SubtreeReindex(unsigned int splitNext) {
  unsigned int chunkSize = 1024;
  unsigned int nChunk = (bagCount + chunkSize - 1) / chunkSize;

  unsigned int chunk;
#pragma omp parallel default(shared) private(chunk)
  {
#pragma omp for schedule(dynamic, 1)
  for (chunk = 0; chunk < nChunk; chunk++) {
    ChunkReindex(bottom->STPath(), splitNext, chunk * chunkSize, (chunk + 1) * chunkSize);
  }
  }
}


/**
   @brief Updates the split/path/pretree state of an extant index based on
   its position in the next level (i.e., left/right/extinct).

   @param stIdx is a subtree-relative index.

   @return void.
 */
void IndexLevel::ChunkReindex(IdxPath *stPath, unsigned int splitNext, unsigned int chunkStart, unsigned int chunkNext) {
  unsigned int chunkEnd = chunkNext > bagCount ? bagCount : chunkNext;
  for (unsigned int stIdx = chunkStart; stIdx < chunkEnd; stIdx++) {
    if (stPath->IsLive(stIdx)) {
      unsigned int pathSucc, ptSucc;
      unsigned int splitIdx = st2Split[stIdx];
      unsigned int splitSucc = indexSet[splitIdx].Offspring(replayExpl->TestBit(stIdx), pathSucc, ptSucc);
      st2Split[stIdx] = splitSucc;
      stPath->Set(stIdx, splitSucc < splitNext ? pathSucc : NodePath::noPath);
      st2PT[stIdx] = ptSucc;
    }
  }
}


/**
   @brief As above, but initializes node-relative mappings for subsequent
   levels.  Employs accumulated state and cannot be parallelized.
 */
void IndexLevel::TransitionReindex(unsigned int splitNext) {
  IdxPath *stPath = bottom->STPath();
  for (unsigned int stIdx = 0; stIdx < bagCount; stIdx++) {
    if (stPath->IsLive(stIdx)) {
      unsigned int pathSucc, idxSucc, ptSucc;
      unsigned int splitIdx = st2Split[stIdx];
      unsigned int splitSucc = indexSet[splitIdx].Offspring(replayExpl->TestBit(stIdx), pathSucc, idxSucc, ptSucc);
      if (splitSucc < splitNext) {
	stPath->SetLive(stIdx, pathSucc, idxSucc);
	rel2ST[idxSucc] = stIdx;
      }
      else {
	stPath->SetExtinct(stIdx);
      }
      st2PT[stIdx] = ptSucc;
    }
  }
}


/**
   @brief Produces next level's index sets, as appropriate, and
   dispatches extinct nodes to pretree frontier.

   @return void.
 */
void IndexLevel::Produce(PreTree *preTree, unsigned int splitNext) {
  bottom->Overlap(samplePred, splitNext, idxLive, nodeRel);
  std::vector<IndexSet> indexNext(splitNext);
  for (auto & iSet : indexSet) {
    iSet.Produce(this, bottom, preTree, indexNext);
  }
  indexSet = std::move(indexNext);
}


/**
   @brief Produces next level's iSets for LH and RH sides of a split.

   @param indexNext is the crescent successor level of index sets.

   @return void, plus output reference parameters.
*/
void IndexSet::Produce(IndexLevel *indexLevel, Bottom *bottom, const PreTree *preTree, std::vector<IndexSet> &indexNext) const {
  if (!terminal) {
    Successor(indexLevel, indexNext, bottom, lhSCount, lhStart, lhExtent, minInfo, preTree->LHId(ptId), leftExpl);
    Successor(indexLevel, indexNext, bottom, sCount - lhSCount, lhStart + lhExtent, extent - lhExtent, minInfo, preTree->RHId(ptId), !leftExpl);
  }
}


/**
   @brief Appends one hand of a split onto next level's iSet list, if
   splitable, otherwise dispatches a terminal iSet.

   @return void.
*/
void IndexSet::Successor(IndexLevel *indexLevel, std::vector<IndexSet> &indexNext, Bottom *bottom, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, bool explHand) const {
  unsigned int succIdx = explHand ? succExpl : succImpl;
  if (succIdx < indexNext.size()) {
    indexNext[succIdx].SuccInit(indexLevel, bottom, succIdx, splitIdx, _sCount, _lhStart, _extent, _minInfo, _ptId, explHand ? sumExpl : sum - sumExpl, explHand ? pathExpl : pathImpl, ctgSum, ctgExpl, explHand);
  }
}


/**
   @brief Initializes index set as a successor node.

   @return void.
 */
void IndexSet::SuccInit(IndexLevel *indexLevel, Bottom *bottom, unsigned int _splitIdx, unsigned int parIdx, unsigned int _sCount, unsigned int _lhStart, unsigned int _extent, double _minInfo, unsigned int _ptId, double _sum, unsigned int _path, const std::vector<SumCount> &_ctgSum, const std::vector<SumCount> &_ctgExpl, bool explHand) {
  Init(_splitIdx, _sCount, _lhStart, _extent, _minInfo, _ptId, _sum, _path, indexLevel->RelBase(_splitIdx), indexLevel->BagCount(), _ctgSum, _ctgExpl, explHand);
  bottom->ReachingPath(splitIdx, parIdx, lhStart, extent, relBase, path);
}


/**
   @brief Visits all live indices, so likely worth parallelizing.
   TODO:  Build categorical sums within Repaly().
 */
void IndexLevel::SumsAndSquares(unsigned int ctgWidth, std::vector<double> &sumSquares, std::vector<double> &ctgSum) {
  unsigned int splitIdx;
  
#pragma omp parallel default(shared) private(splitIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      indexSet[splitIdx].SumsAndSquares(sumSquares[splitIdx], &ctgSum[splitIdx * ctgWidth]);
    }
  }
}


/**
   @brief Sums each category for a node splitable in the upcoming level.

   @param sumSquares accumulates the sum of squares over each category.
   Assumed intialized to zero.

   @param ctgSum records the response sums, by category.  Assumed initialized
   to zero.

   @return void, with side-effected 'unsplitable' state.
   
*/
void IndexSet::SumsAndSquares(double  &sumSquares, double *sumOut) {
  for (unsigned int ctg = 0; ctg < ctgSum.size(); ctg++) {
    unsigned int scSCount;
    ctgSum[ctg].Ref(sumOut[ctg], scSCount);
    sumSquares += sumOut[ctg] * sumOut[ctg];
    if (scSCount == sCount)
      unsplitable = true;
  }
}


void IndexLevel::BlockReplay(IndexSet *iSet, unsigned int predIdx, unsigned int bufIdx, unsigned int blockStart, unsigned int blockExtent) {
    iSet->BlockReplay(samplePred, predIdx, bufIdx, blockStart, blockExtent, replayExpl);
  }


void IndexSet::BlockReplay(SamplePred *samplePred, unsigned int predIdx, unsigned int bufIdx, unsigned int blockStart, unsigned int blockExtent, BV *replayExpl) {
  sumExpl += samplePred->BlockReplay(predIdx, bufIdx, blockStart, blockExtent, replayExpl, ctgExpl);
}


/**
   @brief Sets the prebias fields of all index sets in the level, employing
   SplitPred-specific methods.

   @return void.
 */
void IndexLevel::SetPrebias() {
  for (auto & iSet : indexSet) {
    iSet.SetPrebias(bottom);
  }
}


void IndexSet::SetPrebias(const Bottom *bottom) {
  preBias = bottom->Prebias(splitIdx, sum, sCount);
}
