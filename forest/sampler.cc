// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "predict.h"
#include "leaf.h"
#include "sampler.h"

#include <cmath>

Sampler::Sampler(const vector<double>& yTrain,
		 const class SamplerNux* samples,
		 unsigned int nTree_) :
  nTree(nTree_),
  nObs(yTrain.size()),
  nSamp(nObs), // NO:  set from front end.
  nCtg(0),
  leaf(Leaf::factoryReg(yTrain)),
  bitMatrix(make_unique<BitMatrix>(nTree, nObs)),
  samplerBlock(setExtents(samples)) {
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 const class SamplerNux* samples,
		 unsigned int nTree_,
		 PredictorT nCtg_) :
  nTree(nTree_),
  nObs(yTrain.size()),
  nSamp(nObs), // NO:  set from front end.
  nCtg(nCtg_),
  leaf(Leaf::factoryCtg(yTrain, nCtg)),
  bitMatrix(make_unique<BitMatrix>(nTree, nObs)),
  samplerBlock(setExtents(samples)) {
}


Sampler::Sampler() :
  nTree(0),
  nObs(0),
  nSamp(0),
  nCtg(0),
  bitMatrix(make_unique<BitMatrix>(0, 0)) {
}


BitMatrix* Sampler::getBitMatrix() const {
  return bitMatrix.get();
}


unique_ptr<SamplerBlock> Sampler::setExtents(const SamplerNux* samples) {
  vector<size_t> sampleHeight(nTree);
  vector<size_t> forestIdx;
  size_t leafCount = 0;
  size_t sIdx = 0; // Absolute sample index.

  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT nLeaf = 0;
    size_t leafBase = leafCount;
    IndexT row = 0;
    IndexT sCountTree = 0;
    while (sCountTree != nSamp) {
      sCountTree += samples[sIdx].getSCount();
      row += samples[sIdx].getDelRow();
      bitMatrix->setBit(tIdx, row);
      IndexT leafIdx = samples[sIdx].getLeafIdx();
      nLeaf = max(leafIdx, nLeaf);
      forestIdx.push_back(leafBase + leafIdx);
      sIdx++;
    }
    nLeaf++;
    leafCount += nLeaf;
    sampleHeight[tIdx] = sIdx;
  }
  return  make_unique<SamplerBlock>(samples, move(sampleHeight), forestIdx);
}


SamplerBlock::SamplerBlock(const SamplerNux* samples,
			   const vector<size_t>& height,
			   const vector<size_t>& forestIdx) :
  raw(make_unique<JaggedArrayV<const SamplerNux*, size_t>>(samples, move(height))) {
  sampleExtent = vector<IndexT>(1 + * max_element(forestIdx.begin(), forestIdx.end()));
  for (auto fIdx : forestIdx) {
    sampleExtent[fIdx]++;
  }

  sampleOffset = vector<size_t>(sampleExtent.size());
  size_t countAccum = 0;
  size_t leafIdx = 0;
  for (auto & off : sampleOffset) {
    off = exchange(countAccum, countAccum + sampleExtent[leafIdx++]);
  }
}


vector<IndexT> SamplerBlock::ctgSamples(const Predict* predict,
					const LeafCtg* leaf) const {
  vector<IndexT> ctgCount(leaf->getNCtg() * predict->getScoreCount());
  size_t sIdx = 0; // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < raw->getNMajor(); tIdx++) {
    IndexT row = 0;
    for (; sIdx < getHeight(tIdx); sIdx++) {
      row += getDelRow(sIdx);
      size_t scoreIdx = predict->getScoreIdx(tIdx, getLeafIdx(sIdx));
      ctgCount[scoreIdx * leaf->getNCtg() + leaf->getCtg(row)] += getSCount(sIdx);
    }
  }

  return ctgCount;
}


vector<RankCount> SamplerBlock::countLeafRanks(const class Predict* predict,
					  const vector<IndexT>& row2Rank) const {
  vector<RankCount> rankCount(size());

  vector<size_t> leafTop(sampleOffset.size());
  size_t sIdx = 0; // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < raw->getNMajor(); tIdx++) {
    IndexT row = 0;
    for ( ; sIdx != getHeight(tIdx); sIdx++) {
      row += getDelRow(sIdx);
      size_t leafIdx = predict->getScoreIdx(tIdx, getLeafIdx(sIdx));
      size_t rankIdx = sampleOffset[leafIdx] + leafTop[leafIdx]++;
      rankCount[rankIdx].init(row2Rank[row], getSCount(sIdx));
    }
  }

  return rankCount;
}


void SamplerBlock::dump(const Sampler* sampler,
			vector<vector<size_t> >& rowTree,
			vector<vector<IndexT> >& sCountTree) const {
  if (raw->size() == 0)
    return;

  size_t bagIdx = 0; // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < raw->getNMajor(); tIdx++) {
    IndexT row = 0;
    while (bagIdx != getHeight(tIdx)) {
      row += getDelRow(bagIdx);
      rowTree[tIdx].push_back(row);
      sCountTree[tIdx].push_back(getSCount(bagIdx));
	//	extentTree[tIdx].emplace_back(getExtent(leafIdx)); TODO
    }
  }
}
