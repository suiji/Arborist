// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.cc

   @brief Methods to train leaf components for an entire forest.

   @author Mark Seligman
 */

#include "leaf.h"
#include "sample.h"
#include "bv.h"

#include <algorithm>


/**
   @brief Crescent constructor.
 */
LeafTrain::LeafTrain(unsigned int treeChunk) :
  lbCresc(make_unique<LBCresc>(treeChunk)),
  bbCresc(make_unique<BBCresc>(treeChunk)) {
}


LBCresc::LBCresc(unsigned int nTree) :
  leaf(vector<Leaf>(0)),
  height(vector<size_t>(nTree)),
  treeFloor(0) {
}


BBCresc::BBCresc(unsigned int nTree) :
  bagLeaf(vector<BagLeaf>(0)),
  height(vector<size_t>(nTree)) {
}


LeafTrain::~LeafTrain() {
}


/**
 */
LeafTrainReg::LeafTrainReg(unsigned int treeChunk) :
  LeafTrain(treeChunk) {
}



LeafTrainReg::~LeafTrainReg() {
}


ProbCresc::ProbCresc(unsigned int treeChunk,
                     unsigned int nCtg_,
                     double forestScale_) :
  nCtg(nCtg_),
  treeFloor(0),
  height(vector<size_t>(treeChunk)),
  prob(vector<double>(0)),
  forestScale(forestScale_) {
}

/**
   @brief Constructor for crescent forest.
 */
LeafTrainCtg::LeafTrainCtg(unsigned int treeChunk,
                           unsigned int nCtg_,
                           double scale) :
  LeafTrain(treeChunk),
  probCresc(make_unique<ProbCresc>(treeChunk, nCtg_, scale)),
  nCtg(nCtg_) {
}


LeafTrainCtg::~LeafTrainCtg() {
}


unique_ptr<LeafTrainCtg> LeafTrain::factoryCtg(unsigned int treeChunk,
                                               unsigned int nCtg,
                                               unsigned int nTree,
                                               unsigned int nRow) {
  return make_unique<LeafTrainCtg>(treeChunk, nCtg, 1.0 / (static_cast<double>(nTree)) * nRow);
}

unique_ptr<LeafTrainReg> LeafTrain::factoryReg(unsigned int treeChunk) {
  return make_unique<LeafTrainReg>(treeChunk);
}

void LeafTrain::blockLeaves(const Sample* sample,
                            const vector<unsigned int>& leafMap,
                            unsigned int tIdx) {
  treeInit(sample, leafMap, tIdx); // virtual
  lbCresc->setExtents(leafMap);
  setScores(sample, leafMap); // virtual
  bbCresc->bagLeaves(sample, leafMap);
}


void LeafTrain::treeInit(const Sample* sample,
                         const vector<unsigned int>& leafMap,
                         unsigned int tIdx) {
  lbCresc->treeInit(leafMap, tIdx);
  bbCresc->treeInit(sample, tIdx);
}


void LeafTrainCtg::treeInit(const Sample* sample,
                            const vector<unsigned int>& leafMap,
                            unsigned int tIdx) {
  LeafTrain::treeInit(sample, leafMap, tIdx);
  probCresc->treeInit(lbCresc->getLeafCount(), tIdx);
}
  


// Allocating leaves for current tree.
void LBCresc::treeInit(const vector<unsigned int> &leafMap,
                                 unsigned int tIdx) {
  leafCount = 1 + *max_element(leafMap.begin(), leafMap.end());
  treeFloor = leaf.size();
  height[tIdx] = treeFloor + leafCount;
  Leaf init;
  init.init();
  leaf.insert(leaf.end(), leafCount, init);
}


void LBCresc::setExtents(const vector<unsigned int> &leafMap) {
  for (auto leafIdx : leafMap) {
    leaf[treeFloor + leafIdx].incrExtent();
  }
}

void BBCresc::treeInit(const Sample* sample,
                       unsigned int tIdx) {
  height[tIdx] = bagLeaf.size() + sample->getBagCount();
}


void BBCresc::bagLeaves(const Sample *sample, const vector<unsigned int> &leafMap) {
  // Placing in sIdx order allows row->leaf mapping to be recovered,
  // by appling bag.
  unsigned int sIdx = 0;
  for (auto leafIdx : leafMap) {
    bagLeaf.emplace_back(BagLeaf(leafIdx, sample->getSCount(sIdx++)));
  }
}


void LeafTrainReg::setScores(const Sample* sample, const vector<unsigned int>& leafMap) {
  lbCresc->setScoresReg(sample, leafMap);
}


void LBCresc::setScoresReg(const Sample* sample,
                           const vector<unsigned int>& leafMap) {
  vector<unsigned int> sCount(leafCount); // Per-leaf sample counts.
  fill(sCount.begin(), sCount.end(), 0);

  unsigned int sIdx = 0;
  for (auto leafIdx : leafMap) {
    scoreAccum(leafIdx, sample->getSum(sIdx));
    sCount[leafIdx] += sample->getSCount(sIdx);
    sIdx++;
  }

  auto leafIdx = 0ul;
  for (auto sc : sCount) {
    scoreScale(leafIdx++, 1.0 / sc);
  }
}

void LeafTrainCtg::setScores(const Sample* sample,
                             const vector<unsigned int>& leafMap) {
  probCresc->probabilities(sample, leafMap, lbCresc->getLeafCount());
  lbCresc->setScoresCtg(probCresc.get());
}


void LBCresc::setScoresCtg(const ProbCresc* probCresc) {
  for (unsigned int leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    setScore(leafIdx, probCresc->leafScore(leafIdx));
  }
}


void ProbCresc::probabilities(const Sample* sample,
                              const vector<unsigned int>& leafMap,
                              unsigned int leafCount) {
  vector<double> leafSum(leafCount);
  fill(leafSum.begin(), leafSum.end(), 0.0);

  // Accumulates sample sums by leaf.
  unsigned int sIdx = 0;
  for (auto leafIdx : leafMap) {
    sample->accum(sIdx++, &leafSum[leafIdx], &prob[treeFloor + leafIdx*nCtg]);
  }

  unsigned int leafIdx = 0;
  for (auto sum : leafSum) {
    normalize(leafIdx++, 1.0 / sum);
  }
}

void ProbCresc::normalize(unsigned int leafIdx, double recipSum) {
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    normalize(leafIdx, ctg, recipSum);
  }
}


/**
     @brief Allocates and initializes for all leaf categories in a tree.

     @param leafCount is the number of leaves in the tree.
*/
void ProbCresc::treeInit(unsigned int leafCount, unsigned int tIdx) {
  treeFloor = prob.size();
  height[tIdx] = treeFloor + leafCount * nCtg;
  prob.insert(prob.end(), nCtg * leafCount, 0.0);
}


double ProbCresc::leafScore(unsigned int leafIdx) const {
  double probMax = 0;
  unsigned int argMax = 0;
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    double ctgProb = prob[treeFloor + leafIdx * nCtg + ctg];
    if (ctgProb > probMax) {
      probMax = ctgProb;
      argMax = ctg;
    }
  }

  return argMax + forestScale * probMax;
}


void ProbCresc::dump(double *probOut) const {
  for (size_t i = 0; i < prob.size(); i++) {
    probOut[i] = prob[i];
  }
};



void LeafTrain::cacheNodeRaw(unsigned char leafRaw[]) const {
  lbCresc->dumpRaw(leafRaw);
}


void LBCresc::dumpRaw(unsigned char leafRaw[]) const {
  for (size_t i = 0; i < leaf.size() * sizeof(Leaf); i++) {
    leafRaw[i] = ((unsigned char*) &leaf[0])[i];
  }
}

void LeafTrain::cacheBLRaw(unsigned char blRaw[]) const {
  bbCresc->dumpRaw(blRaw);
}


void BBCresc::dumpRaw(unsigned char blRaw[]) const {
  for (size_t i = 0; i < bagLeaf.size() * sizeof(BagLeaf); i++) {
    blRaw[i] = ((unsigned int*) &bagLeaf[0])[i];
  }
}

void LeafTrainCtg::dumpProb(double probOut[]) const {
  probCresc->dump(probOut);
}


/**
 */
LeafFrameReg::LeafFrameReg(const unsigned int height[],
                 unsigned int nTree_,
                 const Leaf leaf[],
                 const unsigned int bagHeight[],
                 const class BagLeaf bagLeaf[],
                 const double *yTrain_,
                 double meanTrain_,
                 unsigned int rowPredict) :
  LeafFrame(height, nTree_, leaf, bagHeight, bagLeaf),
  yTrain(yTrain_),
  meanTrain(meanTrain_),
  offset(vector<unsigned int>(leafBlock->size())), // leafCount
  defaultScore(MeanTrain()),
  yPred(vector<double>(rowPredict)) {
  leafBlock->setOffsets(offset);
}


/**
   @brief Constructor for trained forest:  vector lengths final.
 */
LeafFrameCtg::LeafFrameCtg(const unsigned int leafHeight[],
                 unsigned int nTree,
                 const class Leaf leaf[],
                 const unsigned int bagHeight[],
                 const class BagLeaf bagLeaf[],
                 const double ctgProb_[],
                 unsigned int ctgTrain_,
                 unsigned int rowPredict,
                 bool doProb) :
  LeafFrame(leafHeight,
       nTree,
       leaf,
       bagHeight,
       bagLeaf),
  ctgTrain(ctgTrain_),
  ctgProb(make_unique<CtgProb>(ctgTrain, nTree, leafHeight, ctgProb_)),
  yPred(vector<unsigned int>(rowPredict)),
  // Can only predict trained categories, so census and
  // probability matrices have 'ctgTrain' columns.
  ctgDefault(ctgProb->ctgDefault()),
  votes(vector<double>(rowPredict * ctgTrain)),
  census(vector<unsigned int>(rowPredict * ctgTrain)),
  prob(vector<double>(doProb ? rowPredict * ctgTrain : 0)) {
  fill(votes.begin(), votes.end(), 0.0);
}


CtgProb::CtgProb(unsigned int ctgTrain,
                 unsigned int nTree,
                 const unsigned int* leafHeight,
                 const double* prob) :
  nCtg(ctgTrain),
  probDefault(vector<double>(nCtg)),
  ctgHeight(scaleHeight(leafHeight, nTree)),
  raw(make_unique<Jagged3<const double*, const unsigned int*> >(nCtg, nTree, &ctgHeight[0], prob)) {
  setDefault();
}


vector<unsigned int> CtgProb::scaleHeight(const unsigned int* leafHeight,
                                          unsigned int nTree) const {
  vector<unsigned int> height(nTree);

  unsigned int i = 0;
  for (auto & ht : height) {
    ht = nCtg * leafHeight[i++];
  }

  return move(height);
}



/**
   @brief Full-forest constructor.
 */
LeafFrame::LeafFrame(const unsigned int* leafHeight,
           unsigned int nTree_,
           const Leaf* leaf,
           const unsigned int* bagHeight,
           const BagLeaf *bagLeaf) :
  nTree(nTree_),
  leafBlock(make_unique<LeafBlock>(nTree, leafHeight, leaf)),
  blBlock(make_unique<BLBlock>(nTree, bagHeight, bagLeaf)),
  noLeaf(leafBlock->size()) { // Greater than all absolute indices.
}


LeafBlock::LeafBlock(unsigned int nTree,
                     const unsigned int* height,
                     const Leaf* leaf) :
  raw(make_unique<JaggedArray<const Leaf*, const unsigned int*> >(nTree, height, leaf)), noLeaf(raw->size()) {
}


BLBlock::BLBlock(unsigned int nTree,
                 const unsigned int* height,
                 const BagLeaf* bagLeaf) :
  raw(make_unique<JaggedArray<const BagLeaf*, const unsigned int*> >(nTree, height, bagLeaf)) {
}
                     


LeafFrame::~LeafFrame() {
}


void LeafBlock::setOffsets(vector<unsigned int>& offset) const {
  unsigned int countAccum = 0;
  for (auto idx = 0ul; idx < raw->size(); idx++) {
    offset[idx] = countAccum;
    countAccum += getExtent(idx);
  }

  // Post-condition:  countAccum == total bag size.
}

/**
   @brief Exporter of BagLeaf vector into per-tree vector of vectors.

   @return void, with output reference parameters.
 */
void LeafFrame::dump(const BitMatrix* baggedRows,
                     vector< vector<unsigned int> > &rowTree,
                     vector< vector<unsigned int> > &sCountTree,
                     vector<vector<double> >& scoreTree,
                     vector<vector<unsigned int> >& extentTree) const {
  blBlock->dump(baggedRows, rowTree, sCountTree);
  leafBlock->dump(scoreTree, extentTree);
}


void LeafBlock::dump(vector<vector<double> >& score,
                     vector<vector<unsigned int> >& extent) const {
  size_t idx = 0;
  for (auto tIdx = 0ul; tIdx < raw->getNMajor(); tIdx++) {
    for (; idx < raw->height[tIdx]; idx++) {
      score[tIdx].push_back(getScore(idx));
      extent[tIdx].push_back(getExtent(idx));
    }
  }
}


void BLBlock::dump(const BitMatrix* baggedRows,
                   vector<vector<unsigned int> >& rowTree,
                   vector<vector<unsigned int> >& sCountTree) const {
  size_t bagIdx = 0;
  for (auto tIdx = 0ul; tIdx < raw->getNMajor(); tIdx++) {
    for (auto row = 0ul; row < baggedRows->getStride(); row++) {
      if (baggedRows->testBit(tIdx, row)) {
        rowTree[tIdx].emplace_back(row);
        sCountTree[tIdx].emplace_back(getSCount(bagIdx++));
      }
    }
  }
}
                                                            

/**
   @brief scores each row in a block independently.

   @return void, with side-effected yPred vector.
 */
void LeafFrameReg::scoreBlock(const unsigned int* predictLeaves,
                              unsigned int rowStart,
                              unsigned int rowEnd) {
  OMPBound blockRow;
  OMPBound blockSup = (OMPBound) (rowEnd - rowStart);

#pragma omp parallel default(shared) private(blockRow)
  {
#pragma omp for schedule(dynamic, 1)
  for (blockRow = 0; blockRow < blockSup; blockRow++) {
    leafBlock->regAcross(&predictLeaves[nTree * blockRow], defaultScore, &yPred[rowStart + blockRow]);
  }
  }
}

void LeafBlock::regAcross(const unsigned int* predictLeaves, double defaultScore, double* yPred) const {
  double score = 0.0;
  unsigned int treesSeen = 0;
  for (unsigned int tIdx = 0; tIdx < nTree(); tIdx++) {
    unsigned int termIdx = predictLeaves[tIdx];
    if (termIdx != noLeaf) {
      treesSeen++;
      score += getScore(tIdx, termIdx);
    }
  }
  *yPred = treesSeen > 0 ? score / treesSeen : defaultScore;
}


// Scores each row independently, in parallel.
void LeafFrameCtg::scoreBlock(const unsigned int* predictLeaves,
                              unsigned int rowStart,
                              unsigned int rowEnd) {
  OMPBound blockRow;
  OMPBound blockSup = (OMPBound) (rowEnd - rowStart);
// TODO:  Recast loop by blocks, to avoid
// false sharing.
#pragma omp parallel default(shared) private(blockRow)
  {
#pragma omp for schedule(dynamic, 1)
  for (blockRow = 0; blockRow < blockSup; blockRow++) {
    leafBlock->ctgAcross(&predictLeaves[nTree * blockRow], ctgDefault, &votes[ctgIdx(rowStart + blockRow, 0)]);
    if (!prob.empty()) {
      ctgProb->probAcross(&predictLeaves[nTree * blockRow], &prob[ctgIdx(rowStart + blockRow, 0)], noLeaf);
    }
  }
  }
}


void LeafBlock::ctgAcross(const unsigned int predictLeaves[],
                          unsigned int ctgDefault,
                          double prediction[]) const {
  unsigned int treesSeen = 0;
  for (unsigned int tIdx = 0; tIdx < nTree(); tIdx++) {
    unsigned int termIdx = predictLeaves[tIdx];
    if (termIdx != noLeaf) {
      treesSeen++;
      double val = getScore(tIdx, termIdx);
      unsigned int ctg = val; // Truncates jittered score for indexing.
      prediction[ctg] += 1 + val - ctg; // 1 plus small jitter.
    }
  }
  if (treesSeen == 0) {
    prediction[ctgDefault] = 1.0; // Other slots all zero.
  }
}


void CtgProb::addLeaf(double* probRow,
                      unsigned int tIdx,
                      unsigned int leafIdx) const {
  auto idxBase = raw->minorOffset(tIdx, leafIdx);
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    probRow[ctg] += raw->getItem(idxBase + ctg);
  }
}


void CtgProb::probAcross(const unsigned int* predictRow,
                         double* probRow,
                         unsigned int noLeaf) const {
  unsigned int treesSeen = 0;
  for (auto tc = 0ul; tc < raw->getNMajor(); tc++) {
    unsigned int termIdx = predictRow[tc];
    if (termIdx != noLeaf) {
      treesSeen++;
      addLeaf(probRow, tc, termIdx);
    }
  }
  if (treesSeen == 0) {
    applyDefault(probRow);
  }
  else {
    double scale = 1.0 / treesSeen;
    for (auto ctg = 0ul; ctg < nCtg; ctg++)
      probRow[ctg] *= scale;
  }
}


/**
   @brief Voting for non-bagged prediction.  Rounds jittered scores to category.

   @return void, with side-effected census.
*/
void LeafFrameCtg::vote() {
  OMPBound rowSup = yPred.size();
  OMPBound row;

#pragma omp parallel default(shared) private(row)
  {
#pragma omp for schedule(dynamic, 1)
  for (row = 0; row < rowSup; row++) {
    unsigned int argMax = ctgTrain;
    double scoreMax = 0.0;
    double *scoreRow = &votes[ctgIdx(row,0)];
    for (unsigned int ctg = 0; ctg < ctgTrain; ctg++) {
      double ctgScore = scoreRow[ctg]; // Jittered vote count.
      if (ctgScore > scoreMax) {
        scoreMax = ctgScore;
        argMax = ctg;
      }
      census[ctgIdx(row, ctg)] = ctgScore; // De-jittered.
    }
    yPred[row] = argMax;
  }
  }
}


/**
 */
void LeafFrameCtg::dump(const BitMatrix *baggedRows,
                        vector<vector<unsigned int> > &rowTree,
                        vector<vector<unsigned int> > &sCountTree,
                        vector<vector<double> > &scoreTree,
                        vector<vector<unsigned int> > &extentTree,
                        vector<vector<double> > &probTree) const {
  LeafFrame::dump(baggedRows, rowTree, sCountTree, scoreTree, extentTree);
  ctgProb->dump(probTree);
}


void CtgProb::dump(vector<vector<double> >& probTree) const {
  size_t off = 0;
  for (auto tIdx = 0ul; tIdx < raw->getNMajor(); tIdx++) {
    for (unsigned int leafIdx = 0; leafIdx < nCtg * raw->height[tIdx]; leafIdx++) {
      probTree[tIdx].push_back(raw->getItem(off++));
    }
  }
}


void CtgProb::setDefault() {
  fill(probDefault.begin(), probDefault.end(), 0.0);

  // Fastest-changing dimension is category.
  for (size_t idx = 0; idx < raw->size(); idx++) {
    probDefault[idx % nCtg] += raw->getItem(idx);
  }

  // Scales by recip leaf count.
  double scale = 1.0 / (raw->size() / nCtg);
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    probDefault[ctg] *= scale;
  }
}


unsigned int CtgProb::ctgDefault() const {
  unsigned int argMax = 0;
  double probMax = 0.0;
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    if (probDefault[ctg] > probMax) {
      probMax = probDefault[ctg];
      argMax = ctg;
    }
  }

  return argMax;  
}


void CtgProb::applyDefault(double *probPredict) const {
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    probPredict[ctg] = probDefault[ctg];
  }
}
