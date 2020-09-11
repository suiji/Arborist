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
                          const vector<IndexT>& leafMap,
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


void BBCresc::bagLeaves(const Sample *sample, const vector<IndexT> &leafMap) {
  unsigned int sIdx = 0;
  for (auto leafIdx : leafMap) {
    bagSample.emplace_back(BagSample(leafIdx, sample->getSCount(sIdx++)));
  }
}


void LFTrainReg::setScores(const Sample* sample, const vector<IndexT>& leafMap) {
  lbCresc->setScoresReg(sample, leafMap);
}


unique_ptr<Sample> LFTrainReg::rootSample(const TrainFrame* frame,
                                          BitMatrix* bag,
                                          unsigned int tIdx) const {
  return Sample::factoryReg(y, frame, bag->BVRow(tIdx).get());
}



void LBCresc::setScoresReg(const Sample* sample,
                           const vector<IndexT>& leafMap) {
  vector<IndexT> sCount(leafCount); // Per-leaf sample counts.
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
                             const vector<IndexT>& leafMap) {
  probCresc->probabilities(sample, leafMap, lbCresc->getLeafCount());
  lbCresc->setScoresCtg(probCresc.get());
}


unique_ptr<Sample> LFTrainCtg::rootSample(const TrainFrame* frame,
                                          BitMatrix* bag,
                                          unsigned int tIdx) const {
  return Sample::factoryCtg(y, frame, &yCtg[0], bag->BVRow(tIdx).get());
}


void LBCresc::setScoresCtg(const ProbCresc* probCresc) {
  for (IndexT leafIdx = 0; leafIdx < leafCount; leafIdx++) {
    setScore(leafIdx, probCresc->leafScore(leafIdx));
  }
}


void ProbCresc::probabilities(const Sample* sample,
                              const vector<IndexT>& leafMap,
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

void ProbCresc::normalize(IndexT leafIdx, double sum) {
  double recipSum = 1.0 / sum;
  double* leafProb = &prob[treeFloor + leafIdx * nCtg];
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    leafProb[ctg] *= recipSum;
  }
}


void ProbCresc::treeInit(IndexT leafCount, unsigned int tIdx) {
  treeFloor = prob.size();
  height[tIdx] = treeFloor + leafCount * nCtg;
  prob.insert(prob.end(), nCtg * leafCount, 0.0);
}


double ProbCresc::leafScore(IndexT leafIdx) const {
  double probMax = 0;
  PredictorT argMax = 0;
  const double* leafProb = &prob[treeFloor + leafIdx * nCtg];
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    double ctgProb = leafProb[ctg];
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



void LFTrain::cacheLeafRaw(unsigned char leafRaw[]) const {
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
