// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.cc

   @brief Methods to train and score leaf components for crescent forest.

   @author Mark Seligman
 */

#include "predict.h"
#include "leaf.h"
#include "sample.h"
#include "sampler.h"
#include "callback.h"

#include <algorithm>

/**
   @brief Crescent constructor.
 */
Leaf::Leaf() {
}


LeafReg::LeafReg(const vector<double>& y_) :
  Leaf(),
  yTrain(y_) {
}


LeafCtg::LeafCtg(const vector<PredictorT>& yCtg_,
		 PredictorT nCtg_) :
  Leaf(),
  yCtg(yCtg_),
  nCtg(nCtg_) {
}


unique_ptr<LeafCtg> Leaf::factoryCtg(const vector<PredictorT>& yCtg,
				     PredictorT nCtg) {
  return make_unique<LeafCtg>(yCtg, nCtg);
}


unique_ptr<LeafReg> Leaf::factoryReg(const vector<double>& yTrain) {
  return make_unique<LeafReg>(yTrain);
}


unique_ptr<Sample> LeafReg::rootSample(const TrainFrame* frame,
				       const vector<double>& yProxy) const {
  return Sample::factoryReg(yTrain, frame);
}


unique_ptr<Sample> LeafCtg::rootSample(const TrainFrame* frame,
				       const vector<double>& yProxy) const {
  return Sample::factoryCtg(yProxy, frame, yCtg);
}


vector<double> LeafReg::scoreTree(const Sample* sample,
				  const vector<IndexT>& leafMap) {
  vector<double> score(1 + *max_element(leafMap.begin(), leafMap.end()));
  vector<IndexT> sCount(score.size());

  IndexT sIdx = 0;
  for (auto leafIdx : leafMap) {
    score[leafIdx] += sample->getSum(sIdx);
    sCount[leafIdx] += sample->getSCount(sIdx);
    sIdx++;
  }

  // Scales scores to per-sample mean.
  IndexT idx = 0ul;
  for (auto sc : sCount) {
    score[idx++] *=  1.0 / sc;
  }

  return score;
}


vector<double> LeafCtg::scoreTree(const Sample* sample,
				  const vector<IndexT>& leafMap) {
  vector<double> score(1 + *max_element(leafMap.begin(), leafMap.end()));
  vector<IndexT> ctgCount = countCtg(score, sample, leafMap);
  vector<double> jitter = CallBack::rUnif(ctgCount.size(), 0.5);
  for (IndexT leafIdx = 0; leafIdx < score.size(); leafIdx++) {
    score[leafIdx] = argMax(leafIdx, ctgCount, jitter);
  }

  return score;
}


vector<PredictorT> LeafCtg::countCtg(const vector<double>& score,
				     const Sample* sample,
				     const vector<IndexT>& leafMap) const {
  vector<IndexT> ctgCount(score.size() * nCtg);
  // Accumulates sample counts by leaf and category.
  IndexT sIdx = 0;
  for (auto leafIdx : leafMap) {
    IndexT ctgIdx = leafIdx * nCtg + sample->getCtg(sIdx);
    ctgCount[ctgIdx]++;
    sIdx++;
  }

  return ctgCount;
}


double LeafCtg::argMax(IndexT leafIdx,
		       const vector<IndexT>& ctgCount,
		       const vector<double>& jitter) {
  IndexT countMax = 0;
  PredictorT argMax = 0;
  const PredictorT* ctgCountLeaf = &ctgCount[leafIdx * nCtg];
  const double* jitterLeaf = &jitter[leafIdx * nCtg];
  for (auto ctg = 0ul; ctg < nCtg; ctg++) {
    IndexT count = ctgCountLeaf[ctg];
    if (count > countMax) {
      countMax = count;
      argMax = ctg;
    }
    else if (count == countMax) {
      if (jitterLeaf[ctg] > jitterLeaf[argMax]) { // Jitter value breaks tie.
	argMax = ctg;
      }
    }
  }

  return argMax + jitterLeaf[argMax];
}


void LeafCtg::ctgBounds(const Predict* predict,
			unsigned int tIdx,
			IndexT leafIdx,
			size_t& start,
			size_t& end) const {
  start = nCtg * predict->getScoreIdx(tIdx, leafIdx);
  end = start + nCtg;
}


void Leaf::cacheScore(double scoreOut[]) const {
  dumpScore(scoreOut);
}


void Leaf::dumpScore(double scoreOut[]) const {
  //  for (size_t i = 0; i < score.size(); i++) {
  //scoreOut[i] = score[i];
}


CtgProb::CtgProb(const Predict* predict,
		 const LeafCtg* leaf,
		 const class Sampler* sampler,
		 bool doProb) :
  nCtg(leaf->getNCtg()),
  ctgCount(sampler->ctgSamples(predict, leaf)),
  ctgHeight(leaf->ctgHeight(predict)),
  raw(make_unique<Jagged3<const IndexT*, const size_t*> >(nCtg, ctgHeight.size(), &ctgHeight[0], &ctgCount[0])),
  probDefault(ctgECDF(predict->scoreHeight.size())),
  probs(vector<double>(doProb ? predict->getNRow() * nCtg : 0)) {
}


vector<size_t> LeafCtg::ctgHeight(const Predict* predict) const {
  vector<size_t> ctgHeight(predict->scoreHeight);
  for (auto & ht : ctgHeight) {
    ht *= nCtg;
  }
  return ctgHeight;
}


void CtgProb::predictRow(const Predict* predict, size_t row) {
  unsigned int nEst = 0;
  vector<IndexT> ctgRow(nCtg);
  for (auto tIdx = 0ul; tIdx < raw->getNMajor(); tIdx++) {
    IndexT termIdx;
    if (predict->isLeafIdx(row, tIdx, termIdx)) {
      nEst++;
      readLeaf(ctgRow, tIdx, termIdx);
    }
  }
  double* probRow = &probs[row * nCtg];
  if (nEst == 0) {
    applyDefault(probRow);
  }
  else {
    size_t sCount = accumulate(ctgRow.begin(), ctgRow.end(), 0ull);
    double scale = 1.0 / sCount;
    for (PredictorT ctg = 0; ctg < nCtg; ctg++)
      probRow[ctg] = ctgRow[ctg] * scale;
  }
}


void CtgProb::readLeaf(vector<IndexT>& ctgRow,
		       unsigned int tIdx,
		       IndexT leafIdx) const {
  size_t idxBase = raw->minorOffset(tIdx, leafIdx);
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    ctgRow[ctg] += raw->getItem(idxBase + ctg);
  }
}


vector<double> CtgProb::ctgECDF(size_t leafCount) {
  // Uses the ECDF as the default distribution.
  vector<PredictorT> ctgTot(nCtg);
  for (size_t base = 0; base < raw->size(); base += nCtg) {
    for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
      ctgTot[ctg] += raw->getItem(base + ctg);
    }
  }

  vector<double> ctgDefault(nCtg);
  double scale = 1.0 / leafCount;
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    ctgDefault[ctg] = ctgTot[ctg] * scale;
  }
  return ctgDefault;
}


void CtgProb::applyDefault(double probPredict[]) const {
  for (PredictorT ctg = 0; ctg < nCtg; ctg++) {
    probPredict[ctg] = probDefault[ctg];
  }
}


void CtgProb::dump() const {
  // TODO
}
