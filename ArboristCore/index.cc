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
#include "splitcand.h"
#include "splitnode.h"
#include "bottom.h"
#include "path.h"
#include "runset.h"
#include "ompthread.h"

#include <numeric>


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
  @brief Sets fields with values used immediately following splitting.

  @return void.
 */
void IndexSet::Init(unsigned int _splitIdx,
                    unsigned int _sCount,
                    unsigned int _lhStart,
                    unsigned int _extent,
                    double _minInfo,
                    unsigned int _ptId,
                    double _sum,
                    unsigned int _path,
                    unsigned int _relBase,
                    unsigned int bagCount,
                    const vector<SumCount> &_ctgSum,
                    const vector<SumCount> &_ctgExpl,
                    bool explHand) {
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
      decr(ctgSum, _ctgExpl);
    }
    ctgExpl = move(vector<SumCount>(ctgSum.size()));

    // Inattainable value.  Reset only when non-terminal:
    succExpl = succImpl = offExpl = offImpl = bagCount;
}

void IndexSet::decr(vector<SumCount> &_ctgSum, const vector<SumCount> &_ctgSub) {
  unsigned i = 0;
  for (auto & sc : _ctgSum) {
    sc.decr(_ctgSub[i++]);
  }
}


/**
   @brief Destructor.

   @return void.
 */
IndexLevel::~IndexLevel() {
  delete replayExpl;
}


IndexSet::IndexSet() :
  splitIdx(0),
  ptId(0),
  lhStart(0),
  extent(0),
  sCount(0),
  sum(0.0),
  minInfo(0.0),
  path(0),
  unsplitable(false),
  sumExpl(0.0){
}


shared_ptr<PreTree> IndexLevel::oneTree(const FrameTrain *frameTrain,
                                        const RowRank* rowRank,
                                        const Sample *sample) {
  auto index = make_unique<IndexLevel>(frameTrain, rowRank, sample);
  return index->levels(frameTrain, sample);
}



/**
   @brief Per-tree constructor.  Sets up root node for level zero.
 */
IndexLevel::IndexLevel(const FrameTrain* frameTrain,
                       const RowRank* rowRank,
                       const Sample* sample) :
  samplePred(sample->predictors()),
  indexSet(vector<IndexSet>(1)),
  bagCount(sample->getBagCount()),
  splitNode(sample->splitNodeFactory(frameTrain)),
  bottom(make_unique<Bottom>(frameTrain, rowRank, bagCount, splitNode.get())),
  nodeRel(false),
  idxLive(bagCount),
  relBase(vector<unsigned int>(1)),
  rel2ST(vector<unsigned int>(bagCount)),
  st2Split(vector<unsigned int>(bagCount)),
  st2PT(vector<unsigned int>(bagCount)),
  replayExpl(new BV(bagCount)) {
  indexSet[0].Init(0, sample->getNSamp(), 0, bagCount, 0.0, 0, sample->getBagSum(), 0, 0, bagCount, sample->getCtgRoot(), sample->getCtgRoot(), true);
  relBase[0] = 0;
  iota(rel2ST.begin(), rel2ST.end(), 0);
  fill(st2Split.begin(), st2Split.end(), 0);
  fill(st2PT.begin(), st2PT.end(), 0);
}


shared_ptr<PreTree> IndexLevel::levels(const FrameTrain *frameTrain,
                                       const Sample* sample) {
  auto stageCount = sample->stage(samplePred.get());
  bottom->rootDef(stageCount);
  shared_ptr<PreTree> preTree = make_shared<PreTree>(frameTrain, bagCount);

  for (unsigned int level = 0; !indexSet.empty(); level++) {
    //cout << "\nLevel " << level << "\n" << endl;
    bottom->levelInit(this);
    auto argMax = bottom->split(samplePred.get(), this);

    unsigned int leafNext, idxMax;
    unsigned int splitNext = splitCensus(argMax, leafNext, idxMax, level + 1 == totLevels);
    consume(preTree.get(), argMax, splitNext, leafNext, idxMax);
    produce(preTree.get(), splitNext);
    bottom->levelClear();
  }

  relFlush();
  preTree->SubtreeFrontier(st2PT);

  return preTree;
}


/**
   @brief Tallies previous level's splitting results.

   @param argMax is a vector of split signatures corresponding to the
   nodes.

   @return count of splitable nodes in the next level.
 */
unsigned int IndexLevel::splitCensus(const vector<SplitCand> &argMax,
                                     unsigned int &leafNext,
                                     unsigned int &idxMax,
                                     bool levelTerminal) {
  this->levelTerminal = levelTerminal;
  unsigned int splitNext, leafThis, idxExtent;
  idxExtent = idxLive; // Previous level's index space.
  leafThis = splitNext = idxLive = idxMax = 0;
  for (auto & iSet : indexSet) {
    iSet.applySplit(argMax);
    iSet.splitCensus(this, leafThis, splitNext, idxLive, idxMax);
  }

  // Restaging is implemented as a patient stable partition.
  //
  leafNext = 2 * (indexSet.size() - leafThis) - splitNext;

  succBase = move(vector<unsigned int>(splitNext + leafNext + leafThis));
  fill(succBase.begin(), succBase.end(), idxExtent); // Inattainable base.

  return splitNext;
}


/**
   @brief Consumes relevant contents of split signature, if any, and accumulates
   leaf and splitting census.

   @param splitNext counts splitable nodes precipitated in the next level.

   @return void.
 */
void IndexSet::splitCensus(IndexLevel *indexLevel,
                           unsigned int &leafThis,
                           unsigned int &splitNext,
                           unsigned int &idxLive,
                           unsigned int &idxMax) {
  if (doesSplit) {
    splitNext += SplitAccum(indexLevel, lhExtent, idxLive, idxMax);
    splitNext += SplitAccum(indexLevel, extent - lhExtent, idxLive, idxMax);
  }
  else {
    leafThis++;
  }
}


/**
     @brief Absorbs parameters of informative splits.

     @return void, with side-effected 'doesSplit' value.
 */
void IndexSet::applySplit(const vector<SplitCand> &argMaxVec) {
  doesSplit = argMaxVec[splitIdx].isInformative(minInfo, lhSCount, lhExtent);
}


/**
    @return count of splitable nodes precipitated in next level:  0/1.
*/
unsigned IndexSet::SplitAccum(class IndexLevel *indexLevel,
                              unsigned int _extent,
                              unsigned int &_idxLive,
                              unsigned int &_idxMax) {
    if (indexLevel->isSplitable(_extent)) {
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
void IndexLevel::consume(PreTree *preTree,
                         const vector<SplitCand> &argMax,
                         unsigned int splitNext,
                         unsigned int leafNext,
                         unsigned int idxMax) {
  preTree->Level(splitNext, leafNext); // Overlap:  two levels co-exist.
  replayExpl->Clear();
  succLive = 0;
  succExtinct = splitNext; // Pseudo-indexing for extinct sets.
  liveBase = 0;
  extinctBase = idxLive;
  for (auto & iSet : indexSet) {
    iSet.consume(this, bottom.get(), preTree, argMax);
  }

  if (nodeRel) {
    nodeReindex();
  }
  else {
    nodeRel = IdxPath::localizes(bagCount, idxMax);
    if (nodeRel) {
      transitionReindex(splitNext);
    }
    else {
      subtreeReindex(splitNext);
    }
  }

  relBase = move(succBase);
}


/**
  @brief Consumes iSet contents into pretree or terminal map.

  @return void.
*/
void IndexSet::consume(IndexLevel *indexLevel, Bottom *bottom, PreTree *preTree, const vector<SplitCand> &argMax) {
  if (doesSplit) {
    nonTerminal(indexLevel, preTree, argMax[splitIdx]);
  }
  else {
    terminal(indexLevel);
  }
}


/**
   @brief Dispatches index set to frontier.

   @return void.
 */
void IndexSet::terminal(IndexLevel *indexLevel) {
  succOnly = indexLevel->idxSucc(extent, ptId, offOnly, true);
}


/**
   @brief Caches state necessary for reindexing and useful subsequently.

   @return void.
 */
void IndexSet::nonTerminal(IndexLevel *indexLevel, PreTree *preTree, const SplitCand &argMax) {
  leftExpl = indexLevel->nonTerminal(preTree, this, argMax);
  ptExpl = leftExpl ? preTree->LHId(ptId) : preTree->RHId(ptId);
  ptImpl = leftExpl ? preTree->RHId(ptId) : preTree->LHId(ptId);
  succExpl = indexLevel->idxSucc(leftExpl ? lhExtent : extent - lhExtent, ptExpl, offExpl);
  succImpl = indexLevel->idxSucc(leftExpl ? extent - lhExtent : lhExtent, ptImpl, offImpl);

  pathExpl = IdxPath::pathNext(path, leftExpl);
  pathImpl = IdxPath::pathNext(path, !leftExpl);
}


/**
   @param sumExpl outputs response sum over explicit hand of the split.

   @return true iff left hand of the split is explicit.
 */
bool IndexLevel::nonTerminal(PreTree *preTree,
                             IndexSet *iSet,
                             const SplitCand &argMax) {
  return nonTerminal(argMax, preTree, iSet, bottom->getRuns());
}


unsigned int IndexLevel::idxSucc(unsigned int extent,
                                 unsigned int ptId,
                                 unsigned int &offOut,
                                 bool predTerminal) {
  unsigned int idxSucc_;
  if (predTerminal || !isSplitable(extent)) { // Pseudo split caches settings.
    idxSucc_ = succExtinct++;
    offOut = extinctBase;
    extinctBase += extent;
  }
  else {
    idxSucc_ = succLive++;
    offOut = liveBase;
    liveBase += extent;
  }
  succBase[idxSucc_] = offOut;

  return idxSucc_;
}


void IndexLevel::nodeReindex() {
  vector<unsigned int> succST(idxLive);
  rel2PT = move(vector<unsigned int>(idxLive));

  OMPBound splitIdx;
#pragma omp parallel default(shared) private(splitIdx)
  {
#pragma omp for schedule(dynamic, 1) 
    for (splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      indexSet[splitIdx].reindex(replayExpl, this, idxLive, succST);
    }
  }
  rel2ST = move(succST);
}


/**
   @brief Node-relative reindexing:  indices contiguous on nodes (index sets).
 */
void IndexSet::reindex(const BV* replayExpl,
                       IndexLevel* index,
                       unsigned int idxLive,
                       vector<unsigned int>& succST) {
  if (!doesSplit) {
    index->relExtinct(relBase, extent, ptId);
  }
  else {
    nontermReindex(replayExpl, index, idxLive, succST);
  }
}


void IndexSet::nontermReindex(const BV *replayExpl,
                              IndexLevel *index,
                              unsigned int idxLive,
                              vector<unsigned int> &succST) {
  unsigned int baseExpl = offExpl;
  unsigned int baseImpl = offImpl;
  for (unsigned int relIdx = relBase; relIdx < relBase + extent; relIdx++) {
    bool expl = replayExpl->testBit(relIdx);
    unsigned int targIdx = expl ? offExpl++ : offImpl++;

    if (targIdx < idxLive) {
      succST[targIdx] = index->relLive(relIdx, targIdx, expl ? pathExpl : pathImpl, expl? baseExpl : baseImpl, expl ? ptExpl : ptImpl);
    }
    else {
      index->relExtinct(relIdx, expl ? ptExpl : ptImpl);
    }
  }
}


/**
   @brief Updates the mapping from live relative indices to associated
   PreTree indices.

   @return corresponding subtree-relative index.
*/
unsigned int IndexLevel::relLive(unsigned int relIdx,
                                 unsigned int targIdx,
                                 unsigned int path,
                                 unsigned int base,
                                 unsigned int ptIdx) {
  unsigned int stIdx = rel2ST[relIdx];
  rel2PT[targIdx] = ptIdx;
  bottom->setLive(relIdx, targIdx, stIdx, path, base);

  return stIdx;
}


/**
   @brief Translates node-relative back to subtree-relative indices on 
   terminatinal node.

   @param relIdx is the node-relative index.

   @param ptId is the pre-tree index of the associated node.

   @return void.
 */
void IndexLevel::relExtinct(unsigned int relIdx, unsigned int ptId) {
  unsigned int stIdx = rel2ST[relIdx];
  st2PT[stIdx] = ptId;
  bottom->setExtinct(relIdx, stIdx);
}


/**
   @brief Subtree-relative reindexing:  indices randomly distributed
   among nodes (i.e., index sets).

   @return void.
*/
void IndexLevel::subtreeReindex(unsigned int splitNext) {
  unsigned int chunkSize = 1024;
  unsigned int nChunk = (bagCount + chunkSize - 1) / chunkSize;

  OMPBound chunk;
#pragma omp parallel default(shared) private(chunk)
  {
#pragma omp for schedule(dynamic, 1)
  for (chunk = 0; chunk < nChunk; chunk++) {
    chunkReindex(bottom->subtreePath(), splitNext, chunk * chunkSize, (chunk + 1) * chunkSize);
  }
  }
}


/**
   @brief Updates the split/path/pretree state of an extant index based on
   its position in the next level (i.e., left/right/extinct).

   @param stIdx is a subtree-relative index.

   @return void.
 */
void IndexLevel::chunkReindex(IdxPath *stPath,
                              unsigned int splitNext,
                              unsigned int chunkStart,
                              unsigned int chunkNext) {
  unsigned int chunkEnd = chunkNext > bagCount ? bagCount : chunkNext;
  for (unsigned int stIdx = chunkStart; stIdx < chunkEnd; stIdx++) {
    if (stPath->isLive(stIdx)) {
      unsigned int pathSucc, ptSucc;
      unsigned int splitIdx = st2Split[stIdx];
      unsigned int splitSucc = indexSet[splitIdx].offspring(replayExpl->testBit(stIdx), pathSucc, ptSucc);
      st2Split[stIdx] = splitSucc;
      stPath->setSuccessor(stIdx, pathSucc, splitSucc < splitNext);
      st2PT[stIdx] = ptSucc;
    }
  }
}


/**
   @brief As above, but initializes node-relative mappings for subsequent
   levels.  Employs accumulated state and cannot be parallelized.
 */
void IndexLevel::transitionReindex(unsigned int splitNext) {
  IdxPath *stPath = bottom->subtreePath();
  for (unsigned int stIdx = 0; stIdx < bagCount; stIdx++) {
    if (stPath->isLive(stIdx)) {
      unsigned int pathSucc, idxSucc, ptSucc;
      unsigned int splitIdx = st2Split[stIdx];
      unsigned int splitSucc = indexSet[splitIdx].offspring(replayExpl->testBit(stIdx), pathSucc, idxSucc, ptSucc);
      if (splitSucc < splitNext) {
        stPath->setLive(stIdx, pathSucc, idxSucc);
        rel2ST[idxSucc] = stIdx;
      }
      else {
        stPath->setExtinct(stIdx);
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
void IndexLevel::produce(const PreTree *preTree, unsigned int splitNext) {
  bottom->Overlap(splitNext, idxLive, nodeRel);
  vector<IndexSet> indexNext(splitNext);
  for (auto & iSet : indexSet) {
    iSet.produce(this, bottom.get(), preTree, indexNext);
  }
  indexSet = move(indexNext);
}


/**
   @brief Produces next level's iSets for LH and RH sides of a split.

   @param indexNext is the crescent successor level of index sets.

   @return void, plus output reference parameters.
*/
void IndexSet::produce(IndexLevel *indexLevel,
                       Bottom *bottom,
                       const PreTree *preTree,
                       vector<IndexSet> &indexNext) const {
  if (doesSplit) {
    successor(indexLevel, indexNext, bottom, lhSCount, lhStart, lhExtent, minInfo, preTree->LHId(ptId), leftExpl);
    successor(indexLevel, indexNext, bottom, sCount - lhSCount, lhStart + lhExtent, extent - lhExtent, minInfo, preTree->RHId(ptId), !leftExpl);
  }
}


/**
   @brief Appends one hand of a split onto next level's iSet list, if
   splitable, otherwise dispatches a terminal iSet.

   @return void.
*/
void IndexSet::successor(IndexLevel *indexLevel,
                         vector<IndexSet> &indexNext,
                         Bottom *bottom,
                         unsigned int _sCount,
                         unsigned int _lhStart,
                         unsigned int _extent,
                         double _minInfo,
                         unsigned int _ptId,
                         bool explHand) const {
  unsigned int succIdx = explHand ? succExpl : succImpl;
  if (succIdx < indexNext.size()) {
    indexNext[succIdx].succInit(indexLevel, bottom, succIdx, splitIdx, _sCount, _lhStart, _extent, _minInfo, _ptId, explHand ? sumExpl : sum - sumExpl, explHand ? pathExpl : pathImpl, ctgSum, ctgExpl, explHand);
  }
}


void IndexSet::succInit(IndexLevel *indexLevel,
                        Bottom *bottom,
                        unsigned int _splitIdx,
                        unsigned int parIdx,
                        unsigned int _sCount,
                        unsigned int _lhStart,
                        unsigned int _extent,
                        double _minInfo,
                        unsigned int _ptId,
                        double _sum,
                        unsigned int _path,
                        const vector<SumCount> &_ctgSum,
                        const vector<SumCount> &_ctgExpl,
                        bool explHand) {
  Init(_splitIdx, _sCount, _lhStart, _extent, _minInfo, _ptId, _sum, _path, indexLevel->RelBase(_splitIdx), indexLevel->getBagCount(), _ctgSum, _ctgExpl, explHand);
  bottom->reachingPath(splitIdx, parIdx, lhStart, extent, relBase, path);
}


/**
   @brief Visits all live indices, so likely worth parallelizing.
   TODO:  Build categorical sums within Replay().
 */
void IndexLevel::sumsAndSquares(unsigned int ctgWidth,
                                vector<double> &sumSquares,
                                vector<double> &ctgSum) {
  OMPBound splitIdx;
#pragma omp parallel default(shared) private(splitIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (splitIdx = 0; splitIdx < indexSet.size(); splitIdx++) {
      indexSet[splitIdx].sumsAndSquares(sumSquares[splitIdx], &ctgSum[splitIdx * ctgWidth]);
    }
  }
}


void IndexSet::sumsAndSquares(double  &sumSquares, double *sumOut) {
  for (unsigned int ctg = 0; ctg < ctgSum.size(); ctg++) {
    unsigned int scSCount;
    ctgSum[ctg].ref(sumOut[ctg], scSCount);
    sumSquares += sumOut[ctg] * sumOut[ctg];
    if (scSCount == sCount)
      unsplitable = true;
  }
}


bool IndexLevel::nonTerminal(const SplitCand &argMax,
                             PreTree *preTree,
                             IndexSet *iSet,
                             const Run *run) const {
  return run->isRun(argMax) ? run->replay(argMax, iSet, preTree, this) : branchNum(argMax, iSet, preTree);
}

bool IndexLevel::branchNum(const SplitCand& argMax,
                          IndexSet* iSet,
                          PreTree* preTree) const {
  preTree->branchNum(argMax, iSet->getPTId());
  iSet->blockReplay(samplePred.get(), argMax, replayExpl);

  return argMax.leftIsExplicit();
}

void IndexSet::blockReplay(SamplePred* samplePred,
                           const SplitCand& argMax,
                           BV* replayExpl) {
  sumExpl += samplePred->blockReplay(argMax, replayExpl, ctgExpl);
}

void IndexSet::blockReplay(SamplePred *samplePred,
                           const SplitCand& argMax,
                           unsigned int blockStart,
                           unsigned int blockExtent,
                           BV *replayExpl) {
  sumExpl += samplePred->blockReplay(argMax, blockStart, blockExtent, replayExpl, ctgExpl);
}


void IndexLevel::blockReplay(IndexSet *iSet,
                             const SplitCand& argMax,
                             unsigned int blockStart,
                             unsigned int blockExtent) const {
  iSet->blockReplay(samplePred.get(), argMax, blockStart, blockExtent, replayExpl);
}
