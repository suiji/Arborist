// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.cc

   @brief Methods to train and score leaf components for an entire forest.

   @author Mark Seligman
 */

#include "leaf.h"
#include "sample.h"
#include "bag.h"
#include "bv.h"
#include "ompthread.h"

#include <algorithm>


LeafFrameReg::LeafFrameReg(const unsigned int height[],
                           unsigned int nTree_,
                           const Leaf leaf[],
                           const unsigned int bagHeight[],
                           const class BagSample bagSample[],
                           const double *yTrain_,
                           size_t rowTrain_,
                           double meanTrain_,
                           unsigned int rowPredict) :
  LeafFrame(height, nTree_, leaf, bagHeight, bagSample),
  yTrain(yTrain_),
  rowTrain(rowTrain_),
  meanTrain(meanTrain_),
  offset(leafBlock->setOffsets()), // leafCount
  defaultScore(MeanTrain()),
  yPred(vector<double>(rowPredict)) {
}


LeafFrameCtg::LeafFrameCtg(const unsigned int leafHeight[],
                 unsigned int nTree,
                 const class Leaf leaf[],
                 const unsigned int bagHeight[],
                 const class BagSample bagSample[],
                 const double ctgProb_[],
                 unsigned int ctgTrain_,
                 unsigned int rowPredict,
                 bool doProb) :
  LeafFrame(leafHeight,
       nTree,
       leaf,
       bagHeight,
       bagSample),
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

  return height;
}


LeafFrame::LeafFrame(const unsigned int* leafHeight,
           unsigned int nTree_,
           const Leaf* leaf,
           const unsigned int* bagHeight,
           const BagSample *bagSample) :
  nTree(nTree_),
  leafBlock(make_unique<LeafBlock>(nTree, leafHeight, leaf)),
  blBlock(make_unique<BLBlock>(nTree, bagHeight, bagSample)),
  noLeaf(leafBlock->size()) { // Greater than all absolute indices.
}


LeafBlock::LeafBlock(unsigned int nTree,
                     const unsigned int* height,
                     const Leaf* leaf) :
  raw(make_unique<JaggedArray<const Leaf*, const unsigned int*> >(nTree, height, leaf)), noLeaf(raw->size()) {
}


BLBlock::BLBlock(unsigned int nTree,
                 const unsigned int* height,
                 const BagSample* bagSample) :
  raw(make_unique<JaggedArray<const BagSample*, const unsigned int*> >(nTree, height, bagSample)) {
}
                     

vector<size_t> LeafBlock::setOffsets() const {
  vector<size_t> offset(raw->size());
  unsigned int countAccum = 0;
  unsigned int idx = 0;
  for (auto & off : offset) {
    off = countAccum;
    countAccum += getExtent(idx++);
  }

  return offset;
  // Post-condition:  countAccum == total bag size.
}


void LeafFrame::dump(const Bag* bag,
                     vector< vector<size_t> > &rowTree,
                     vector< vector<unsigned int> > &sCountTree,
                     vector<vector<double> >& scoreTree,
                     vector<vector<unsigned int> >& extentTree) const {
  blBlock->dump(bag, rowTree, sCountTree);
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


void BLBlock::dump(const Bag* bag,
                   vector<vector<size_t> >& rowTree,
                   vector<vector<unsigned int> >& sCountTree) const {
  size_t bagIdx = 0;
  const BitMatrix* baggedRows(bag->getBitMatrix());
  for (auto tIdx = 0ul; tIdx < raw->getNMajor(); tIdx++) {
    for (auto row = 0ul; row < baggedRows->getStride(); row++) {
      if (baggedRows->testBit(tIdx, row)) {
        rowTree[tIdx].emplace_back(row);
        sCountTree[tIdx].emplace_back(getSCount(bagIdx++));
      }
    }
  }
}
                                                            

vector<RankCount> LeafFrameReg::setRankCount(const BitMatrix* baggedRows,
                               const vector<unsigned int>& row2Rank) const {
  vector<RankCount> rankCount(blBlock->size());
  if (baggedRows->isEmpty())
    return rankCount; // Short circuits with empty vector.

  vector<unsigned int> leafSeen(leafCount());
  fill(leafSeen.begin(), leafSeen.end(), 0);
  unsigned int bagIdx = 0;  // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    for (unsigned int row = 0; row < rowTrain; row++) {
      if (baggedRows->testBit(tIdx, row)) {
        unsigned int leafAbs = getLeafAbs(tIdx, bagIdx);
        unsigned int sIdx = offset[leafAbs] + leafSeen[leafAbs]++;
        rankCount[sIdx].init(row2Rank[row], getSCount(bagIdx));
        bagIdx++;
      }
    }
  }

  return rankCount;
}


void LeafFrameReg::scoreBlock(const unsigned int* predictLeaves,
                              size_t rowStart,
                              size_t extent) {
  OMPBound blockSup = (OMPBound) extent;

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound blockRow = 0; blockRow < blockSup; blockRow++) {
    leafBlock->scoreAcross(&predictLeaves[nTree * blockRow], defaultScore, &yPred[rowStart + blockRow]);
  }
  }
}

void LeafBlock::scoreAcross(const unsigned int* predictLeaves, double defaultScore, double* yPred) const {
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
                              size_t rowStart,
                              size_t extent) {
  OMPBound blockSup = (OMPBound) extent;
// TODO:  Recast loop by blocks, to avoid
// false sharing.
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound blockRow = 0; blockRow < blockSup; blockRow++) {
    leafBlock->scoreAcross(&predictLeaves[nTree * blockRow], ctgDefault, &votes[ctgIdx(rowStart + blockRow, 0)]);
    if (!prob.empty()) {
      ctgProb->probAcross(&predictLeaves[nTree * blockRow], &prob[ctgIdx(rowStart + blockRow, 0)], noLeaf);
    }
  }
  }
}


void LeafBlock::scoreAcross(const unsigned int predictLeaves[],
                          unsigned int ctgDefault,
                          double yCtg[]) const {
  unsigned int treesSeen = 0;
  for (unsigned int tIdx = 0; tIdx < nTree(); tIdx++) {
    unsigned int termIdx = predictLeaves[tIdx];
    if (termIdx != noLeaf) {
      treesSeen++;
      double val = getScore(tIdx, termIdx);
      unsigned int ctg = floor(val); // Truncates jittered score for indexing.
      yCtg[ctg] += (1.0 + val) - ctg; // 1 plus small jitter.
    }
  }
  if (treesSeen == 0) {
    yCtg[ctgDefault] = 1.0; // Other slots all zero.
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

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound row = 0; row < rowSup; row++) {
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


void LeafFrameCtg::dump(const Bag* bag,
                        vector<vector<size_t> > &rowTree,
                        vector<vector<unsigned int> > &sCountTree,
                        vector<vector<double> > &scoreTree,
                        vector<vector<unsigned int> > &extentTree,
                        vector<vector<double> > &probTree) const {
  LeafFrame::dump(bag, rowTree, sCountTree, scoreTree, extentTree);
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

/**
   @brief Crescent constructor.
 */
LFTrain::LFTrain(const double* y_, unsigned int treeChunk) :
  y(y_),
  lbCresc(make_unique<LBCresc>(treeChunk)),
  bbCresc(make_unique<BBCresc>(treeChunk)) {
}


LBCresc::LBCresc(unsigned int nTree) :
  leaf(vector<Leaf>(0)),
  height(vector<size_t>(nTree)),
  treeFloor(0) {
}


BBCresc::BBCresc(unsigned int nTree) :
  bagSample(vector<BagSample>(0)),
  height(vector<size_t>(nTree)) {
}


LFTrainReg::LFTrainReg(const double* y,
                       unsigned int treeChunk) :
  LFTrain(y, treeChunk) {
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


LFTrainCtg::LFTrainCtg(const unsigned int* yCtg_,
                       const double* proxy,
                       unsigned int treeChunk,
                       unsigned int nCtg,
                       double scale) :
  LFTrain(proxy, treeChunk),
  yCtg(yCtg_),
  probCresc(make_unique<ProbCresc>(treeChunk, nCtg, scale)) {
}


unique_ptr<LFTrainCtg> LFTrain::factoryCtg(const unsigned int* feResponse,
                                           const double* feProxy,
                                           unsigned int treeChunk,
                                           unsigned int nRow,
                                           unsigned int nCtg,
                                           unsigned int nTree) {
  return make_unique<LFTrainCtg>(feResponse, feProxy, treeChunk, nCtg, 1.0 / (static_cast<double>(nTree) * nRow));
}

unique_ptr<LFTrainReg> LFTrain::factoryReg(const double* feResponse,
                                           unsigned int treeChunk) {
  return make_unique<LFTrainReg>(feResponse, treeChunk);
}


void LFTrain::blockLeaves(const Sample* sample,
                          const vector<unsigned int>& leafMap,
                          unsigned int tIdx) {
  treeInit(sample, leafMap, tIdx); // virtual
  lbCresc->setExtents(leafMap);
  setScores(sample, leafMap); // virtual
  bbCresc->bagLeaves(sample, leafMap);
}


void LFTrain::treeInit(const Sample* sample,
                       const vector<unsigned int>& leafMap,
                       unsigned int tIdx) {
  lbCresc->treeInit(leafMap, tIdx);
  bbCresc->treeInit(sample, tIdx);
}


void LFTrainCtg::treeInit(const Sample* sample,
                            const vector<unsigned int>& leafMap,
                            unsigned int tIdx) {
  LFTrain::treeInit(sample, leafMap, tIdx);
  probCresc->treeInit(lbCresc->getLeafCount(), tIdx);
}
  


void LBCresc::treeInit(const vector<unsigned int> &leafMap,
                                 unsigned int tIdx) {
  leafCount = 1 + *max_element(leafMap.begin(), leafMap.end());
  treeFloor = leaf.size();
  height[tIdx] = treeFloor + leafCount;
  Leaf init;
  leaf.insert(leaf.end(), leafCount, init);
}


void LBCresc::setExtents(const vector<unsigned int> &leafMap) {
  for (auto leafIdx : leafMap) {
    leaf[treeFloor + leafIdx].incrExtent();
  }
}

void BBCresc::treeInit(const Sample* sample,
                       unsigned int tIdx) {
  height[tIdx] = bagSample.size() + sample->getBagCount();
}


void BBCresc::bagLeaves(const Sample *sample, const vector<unsigned int> &leafMap) {
  unsigned int sIdx = 0;
  for (auto leafIdx : leafMap) {
    bagSample.emplace_back(BagSample(leafIdx, sample->getSCount(sIdx++)));
  }
}


void LFTrainReg::setScores(const Sample* sample, const vector<unsigned int>& leafMap) {
  lbCresc->setScoresReg(sample, leafMap);
}


unique_ptr<Sample> LFTrainReg::rootSample(const SummaryFrame* frame,
                                          BitMatrix* bag,
                                          unsigned int tIdx) const {
  return Sample::factoryReg(y, frame, bag->BVRow(tIdx).get());
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

void LFTrainCtg::setScores(const Sample* sample,
                             const vector<unsigned int>& leafMap) {
  probCresc->probabilities(sample, leafMap, lbCresc->getLeafCount());
  lbCresc->setScoresCtg(probCresc.get());
}


unique_ptr<Sample> LFTrainCtg::rootSample(const SummaryFrame* frame,
                                          BitMatrix* bag,
                                          unsigned int tIdx) const {
  return Sample::factoryCtg(y, frame, &yCtg[0], bag->BVRow(tIdx).get());
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
    sample->accum(sIdx++, leafSum[leafIdx], &prob[treeFloor + leafIdx*nCtg]);
  }

  unsigned int leafIdx = 0;
  for (auto sum : leafSum) {
    normalize(leafIdx++, sum);
  }
}

void ProbCresc::normalize(unsigned int leafIdx, double sum) {
  double recipSum = 1.0 / sum;
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    prob[treeFloor + leafIdx*nCtg + ctg] *= recipSum;
  }
}


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
  // Integer component of score is argMax.
  // Fractional part is scaled probability value.
  return argMax + forestScale * probMax;
}


void ProbCresc::dump(double *probOut) const {
  for (size_t i = 0; i < prob.size(); i++) {
    probOut[i] = prob[i];
  }
}



void LFTrain::cacheNodeRaw(unsigned char leafRaw[]) const {
  lbCresc->dumpRaw(leafRaw);
}


void LBCresc::dumpRaw(unsigned char leafRaw[]) const {
  for (size_t i = 0; i < leaf.size() * sizeof(Leaf); i++) {
    leafRaw[i] = ((unsigned char*) &leaf[0])[i];
  }
}

void LFTrain::cacheBLRaw(unsigned char blRaw[]) const {
  bbCresc->dumpRaw(blRaw);
}


void BBCresc::dumpRaw(unsigned char blRaw[]) const {
  for (size_t i = 0; i < bagSample.size() * sizeof(BagSample); i++) {
    blRaw[i] = ((unsigned char*) &bagSample[0])[i];
  }
}

void LFTrainCtg::dumpWeight(double probOut[]) const {
  probCresc->dump(probOut);
}
