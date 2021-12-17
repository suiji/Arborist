// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include "predict.h"
#include "leaf.h"
#include "sampler.h"

#include <cmath>

PackedT RankCount::rankMask = 0;
unsigned int RankCount::rightBits = 0;

unsigned int SamplerNux::delWidth = 0;
PackedT SamplerNux::delMask = 0;
PackedT SamplerNux::leafMask = 0;
unsigned int SamplerNux::rightBits = 0;

Sampler::Sampler(const vector<double>& yTrain,
		 bool nuxSamples_,
		 IndexT nSamp_,
		 unsigned int treeChunk,
		 bool bagging_) :
  nTree(treeChunk),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  nCtg(0),
  bagging(bagging_),
  nuxSamples(nuxSamples_),
  leaf(Leaf::factoryReg(yTrain)),
  bagMatrix((bagging && !nuxSamples) ? make_unique<BitMatrix>(nTree, nObs) : make_unique<BitMatrix>(0,0)),
  samplerBlock(nullptr),
  tIdx(0) {
  RankCount::setMasks(nObs);
  SamplerNux::setMasks(nObs, nSamp);
}


Sampler::Sampler(const vector<double>& yTrain,
		 bool nuxSamples_,
		 unsigned char* samples,
		 IndexT nSamp_,
		 unsigned int nTree_,
		 bool bagging_) :
  nTree(nTree_),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  nCtg(0),
  bagging(bagging_),
  nuxSamples(nuxSamples_),
  leaf(Leaf::factoryReg(yTrain)),
  bagMatrix(bagRaw(samples, nuxSamples, bagging, nTree, nObs)) {
  RankCount::setMasks(nObs);
  SamplerNux::setMasks(nObs, nSamp);
  samplerBlock = readRaw(samples);
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 bool nuxSamples_,
		 IndexT nSamp_,
		 unsigned int treeChunk,
		 PredictorT nCtg_,
		 const vector<double>& classWeight,
		 bool bagging_) :
  nTree(treeChunk),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  nCtg(nCtg_),
  bagging(bagging_),
  nuxSamples(nuxSamples_),
  leaf(Leaf::factoryCtg(yTrain, nCtg, classWeight)),
  bagMatrix((bagging && !nuxSamples) ? make_unique<BitMatrix>(nTree, nObs) : make_unique<BitMatrix>(0,0)),
  samplerBlock(nullptr),
  tIdx(0) {
  RankCount::setMasks(nObs);
  SamplerNux::setMasks(nObs, nSamp);
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 bool nuxSamples_,
		 unsigned char* samples,
		 IndexT nSamp_,
		 unsigned int nTree_,
		 PredictorT nCtg_,
		 bool bagging_) :
  nTree(nTree_),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  nCtg(nCtg_),
  bagging(bagging_),
  nuxSamples(nuxSamples_),
  leaf(Leaf::factoryCtg(yTrain, nCtg)),
  bagMatrix(bagRaw(samples, nuxSamples, bagging, nTree, nObs)) {
  RankCount::setMasks(nObs);
  SamplerNux::setMasks(nObs, nSamp);
  samplerBlock = readRaw(samples);
}


unique_ptr<BitMatrix> Sampler::bagRaw(unsigned char* rawSamples,
				      bool nuxSamples,
				      bool bagging,
				      unsigned int nTree, IndexT nObs) {
  if (bagging) {
    return nuxSamples ? make_unique<BitMatrix>(nTree, nObs) : make_unique<BitMatrix>(reinterpret_cast<unsigned int*>(rawSamples), nTree, nObs);
  }
  else {
    return  make_unique<BitMatrix>(0,0);
  }
}


unique_ptr<SamplerBlock> Sampler::readRaw(unsigned char* rawSamples) {
  if (!nuxSamples)
    return nullptr;

  SamplerNux* samples = reinterpret_cast<SamplerNux*>(rawSamples);
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
      if (bagging)
	bagMatrix->setBit(tIdx, row);
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


vector<IndexT> SamplerBlock::countLeafCtg(const Predict* predict,
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


void Sampler::rootSample(const class TrainFrame* frame) {
  sample = leaf->rootSample(frame, this);
}


Sample* Sampler::getSample() const {
  return sample.get();
}


void Sampler::blockSamples(const vector<IndexT>& leafMap) {
  if (!bagMatrix->isEmpty()) { // Thin, but bagging.
    IndexT row = 0;
    for (IndexT sIdx = 0; sIdx < leafMap.size(); sIdx++) {
      row += sample->getDelRow(sIdx);
      bagMatrix->setBit(tIdx, row);
    }
  }
  else {
    IndexT sIdx = 0;
    for (auto leafIdx : leafMap) {
      sbCresc.emplace_back(sample->getDelRow(sIdx), leafIdx, sample->getSCount(sIdx));
      sIdx++;
    }
  }
  tIdx++;
}


size_t Sampler::getBlockBytes() const {
  if (nuxSamples) {
    return sbCresc.size() * sizeof(SamplerNux);
  }
  else {
    return bagMatrix->getNSlot() * sizeof(unsigned int);
  }
}


vector<double> Sampler::scoreTree(const vector<IndexT>& leafMap) const {
  return leaf->scoreTree(sample.get(), leafMap);
}


void Sampler::dumpRaw(unsigned char outRaw[]) const {
  if (nuxSamples)
    dumpNuxRaw(outRaw);
  else
    bagMatrix->dumpRaw(outRaw);
}


void Sampler::dumpNuxRaw(unsigned char snRaw[]) const {
  for (size_t i = 0; i < sbCresc.size() * sizeof(SamplerNux); i++) {
    snRaw[i] = reinterpret_cast<const unsigned char*>(&sbCresc[0])[i];
  }
}
