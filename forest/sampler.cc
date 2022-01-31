// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include "predict.h"
#include "sampler.h"
#include "samplemap.h"
#include "pretree.h"
#include "callback.h"

#include <cmath>

PackedT RankCount::rankMask = 0;
unsigned int RankCount::rightBits = 0;

PackedT SamplerNux::delMask = 0;
unsigned int SamplerNux::rightBits = 0;


unique_ptr<Sampler> Sampler::trainReg(const vector<double>& yTrain,
				      bool nuxSamples,
				      IndexT nSamp,
				      unsigned int treeChunk) {
  RankCount::setMasks(yTrain.size());
  SamplerNux::setMasks(yTrain.size());
  return make_unique<Sampler>(yTrain, nuxSamples, nSamp, treeChunk);
}


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
  samplerBlock(nullptr) {
}


unique_ptr<Sampler> Sampler::predictReg(const vector<double>& yTrain,
					bool nuxSamples,
					const unsigned char samples[],
					IndexT nSamp,
					unsigned int nTree,
					const double extentNum[],
					const double indexNum[],
					bool bagging) {
  RankCount::setMasks(yTrain.size());
  SamplerNux::setMasks(yTrain.size());
  return make_unique<Sampler>(yTrain, nuxSamples, samples, nSamp, nTree, extentNum, indexNum, bagging);
}


Sampler::Sampler(const vector<double>& yTrain,
		 bool nuxSamples_,
		 const unsigned char samples[],
		 IndexT nSamp_,
		 unsigned int nTree_,
		 const double extentNum[],
		 const double indexNum[],
		 bool bagging_) :
  nTree(nTree_),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  nCtg(0),
  bagging(bagging_),
  nuxSamples(nuxSamples_),
  bagCount(countSamples(samples)),
  extent(unpackExtent(extentNum)),
  index(unpackIndex(indexNum)),
  leaf(Leaf::factoryReg(yTrain)),
  bagMatrix(bagRaw(samples, nuxSamples, bagging, nTree, nObs)),
  samplerBlock(readRaw(samples)) {
  samplerBlock->bagRows(bagMatrix.get(), nuxSamples);
}


vector<size_t> Sampler::countSamples(const unsigned char rawSamples[]) const {
  vector<size_t> sampleCount(nTree);
  if (!nuxSamples)
    return sampleCount;

  const SamplerNux* samples = reinterpret_cast<const SamplerNux*>(rawSamples);
  size_t sIdx = 0; // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT sIdxStart = sIdx;
    IndexT sCountTree = 0;
    while (sCountTree != nSamp) {
      sCountTree += samples[sIdx].getSCount();
      sIdx++;
    }
    sampleCount[tIdx] = sIdx - sIdxStart;
  }

  return sampleCount;
}


vector<vector<size_t>> Sampler::unpackExtent(const double extentNum[]) const {
  vector<vector<size_t>> unpacked(nTree);
  if (extentNum == nullptr)
    return unpacked;

  size_t idx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    size_t extentTree = 0;
    while (extentTree < bagCount[tIdx]) {
      size_t extentLeaf = extentNum[idx++];
      unpacked[tIdx].push_back(extentLeaf);
      extentTree += extentLeaf;
    }
  }
  return unpacked;
}


vector<vector<vector<size_t>>> Sampler::unpackIndex(const double numVal[]) const {
  vector<vector<vector<size_t>>> unpacked(nTree);
  if (numVal == nullptr)
    return unpacked;

  size_t idx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    unpacked[tIdx] = vector<vector<size_t>>(getLeafCount(tIdx)); // # leaves in tree
    for (size_t leafIdx = 0; leafIdx < unpacked[tIdx].size(); leafIdx++) { // # sample indices in leaf.
      vector<size_t> unpackedLeaf(extent[tIdx][leafIdx]);
      for (size_t slot = 0; slot < unpackedLeaf.size(); slot++) {
	unpackedLeaf[slot] = numVal[idx];
	idx++;
      }
      unpacked[tIdx][leafIdx] = unpackedLeaf;
    }
  }
  return unpacked;
}


unique_ptr<Sampler> Sampler::trainCtg(const vector<PredictorT>& yTrain,
				      bool nuxSamples,
				      IndexT nSamp,
				      unsigned int treeChunk,
				      PredictorT nCtg,
				      const vector<double>& classWeight) {
  RankCount::setMasks(yTrain.size());
  SamplerNux::setMasks(yTrain.size());
  return make_unique<Sampler>(yTrain, nuxSamples, nSamp, treeChunk, nCtg, classWeight);
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
  samplerBlock(nullptr) {
}


unique_ptr<Sampler>  Sampler::predictCtg(const vector<PredictorT>& yTrain,
					 bool nuxSamples,
					 const unsigned char samples[],
					 IndexT nSamp,
					 unsigned int nTree,
					 const double extentNum[],
					 const double indexNum[],
					 PredictorT nCtg,
					 bool bagging) {
  RankCount::setMasks(yTrain.size());
  SamplerNux::setMasks(yTrain.size());
  return make_unique<Sampler>(yTrain, nuxSamples, samples, nSamp, nTree, extentNum, indexNum, nCtg, bagging);
}


Sampler::Sampler(const vector<PredictorT>& yTrain,
		 bool nuxSamples_,
		 const unsigned char samples[],
		 IndexT nSamp_,
		 unsigned int nTree_,
		 const double extentNum[],
		 const double indexNum[],
		 PredictorT nCtg_,
		 bool bagging_) :
  nTree(nTree_),
  nObs(yTrain.size()),
  nSamp(nSamp_),
  nCtg(nCtg_),
  bagging(bagging_),
  nuxSamples(nuxSamples_),
  bagCount(countSamples(samples)),
  extent(unpackExtent(extentNum)),
  index(unpackIndex(indexNum)),
  leaf(Leaf::factoryCtg(yTrain, nCtg)),
  bagMatrix(bagRaw(samples, nuxSamples, bagging, nTree, nObs)),
  samplerBlock(readRaw(samples)) {
  samplerBlock->bagRows(bagMatrix.get(), nuxSamples);
}


unique_ptr<BitMatrix> Sampler::bagRaw(const unsigned char rawSamples[],
				      bool nuxSamples,
				      bool bagging,
				      unsigned int nTree,
				      IndexT nObs) {
  if (bagging) {
    return nuxSamples ? make_unique<BitMatrix>(nTree, nObs) : make_unique<BitMatrix>(reinterpret_cast<const BVSlotT*>(rawSamples), nTree, nObs);
  }
  else {
    return  make_unique<BitMatrix>(0,0);
  }
}


unique_ptr<SamplerBlock> Sampler::readRaw(const unsigned char rawSamples[]) {
  if (!nuxSamples)
    return nullptr;

  const SamplerNux* samples = reinterpret_cast<const SamplerNux*>(rawSamples);
  vector<size_t> sampleHeight(nTree);
  partial_sum(bagCount.begin(), bagCount.end(), sampleHeight.begin());
  return  make_unique<SamplerBlock>(this, samples, move(sampleHeight));
}


SamplerBlock::SamplerBlock(const Sampler* sampler,
			   const SamplerNux* samples,
			   const vector<size_t>& height) :
  raw(make_unique<JaggedArrayV<const SamplerNux*, size_t>>(samples, move(height))) {
}


void SamplerBlock::bagRows(BitMatrix* bagMatrix,
			   bool nuxSamples) {
  if (bagMatrix->isEmpty() || !nuxSamples)
    return;

  size_t sIdx = 0; // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < raw->getNMajor(); tIdx++) {
    IndexT row = 0;
    for (; sIdx != getHeight(tIdx); sIdx++) {
      row += getDelRow(sIdx);
      bagMatrix->setBit(tIdx, row);
    }
  }
}

// Move to Leaf:
vector<vector<vector<size_t>>> SamplerBlock::countLeafCtg(const Sampler* sampler,
							  const LeafCtg* leaf) const {
  vector<vector<vector<size_t>>> ctgCount(sampler->getNTree());
  PredictorT nCtg = sampler->getNCtg();
  size_t treeIdx = 0; // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < raw->getNMajor(); tIdx++) {
    IndexT row = 0;
    vector<PredictorT> sIdx2Ctg(sampler->getBagCount(tIdx));
    for (IndexT sIdx = 0; sIdx != sIdx2Ctg.size(); sIdx++) {
      row += getDelRow(treeIdx + sIdx);
      sIdx2Ctg[sIdx] = leaf->getCtg(row);
    }
    size_t leafIdx = 0;
    ctgCount[tIdx] = vector<vector<size_t>>(sampler->getLeafCount(tIdx));
    for (vector<size_t> sIdxVec : sampler->getIndices(tIdx)) {
      ctgCount[tIdx][leafIdx] = vector<size_t>(sIdxVec.size() * nCtg);
      for (size_t sIdx : sIdxVec) {
	PredictorT ctg = sIdx2Ctg[sIdx];
	ctgCount[tIdx][leafIdx][ctg] += getSCount(treeIdx + sIdx);
      }
      leafIdx++;
    }
    treeIdx += sampler->getBagCount(tIdx);
  }

  return ctgCount;
}

// Move to Leaf:
vector<vector<vector<RankCount>>> SamplerBlock::alignRanks(const Sampler* sampler,
							   const vector<IndexT>& row2Rank) const {
  vector<vector<vector<RankCount>>> rankCount(sampler->getNTree());
  size_t treeIdx = 0;
  for (unsigned int tIdx = 0; tIdx < raw->getNMajor(); tIdx++) {
    IndexT row = 0;
    vector<size_t> sIdx2Rank(sampler->getBagCount(tIdx));
    for (IndexT sIdx = 0 ; sIdx != sIdx2Rank.size(); sIdx++) {
      row += getDelRow(treeIdx + sIdx);
      sIdx2Rank[sIdx] = row2Rank[row];
    }
    size_t leafIdx = 0;
    rankCount[tIdx] = vector<vector<RankCount>>(sampler->getLeafCount(tIdx));
    for (vector<size_t> sIdxVec : sampler->getIndices(tIdx)) {
      rankCount[tIdx][leafIdx] = vector<RankCount>(sIdxVec.size());
      size_t idx = 0;
      for (size_t sIdx : sIdxVec) {
	rankCount[tIdx][leafIdx][idx++].init(sIdx2Rank[sIdx], getSCount(treeIdx + sIdx));
      }
      leafIdx++;
    }
    treeIdx += sampler->getBagCount(tIdx);
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


unique_ptr<Sample> Sampler::rootSample(const class TrainFrame* frame,
				       unsigned int tIdx) {
  sCountRow = countSamples(nObs, nSamp);
  if (!bagMatrix->isEmpty()) { // Thin, but bagging.
    for (IndexT row = 0; row < nObs; row++) {
      if (sCountRow[row] > 0) {
	bagMatrix->setBit(tIdx, row);
      }
    }
  }
  else {
    IndexT rowPrev = 0;
    for (IndexT row = 0; row < nObs; row++) {
      if (sCountRow[row] > 0) {
        sbCresc.emplace_back(row - exchange(rowPrev, row), sCountRow[row]);
      }
    }
  }
  return leaf->rootSample(frame, this); //, sbCresc[tIdx]
}


// Sample counting is sensitive to locality.  In the absence of
// binning, access is random.  Larger bins improve locality, but
// performance begins to degrade when bin size exceeds available
// cache.
vector<IndexT> Sampler::countSamples(IndexT nRow,
				     IndexT nSamp) {
  vector<IndexT> sc(nRow);
  vector<IndexT> idx(CallBack::sampleRows(nSamp));
  if (binIdx(sc.size()) > 0) {
    idx = binIndices(idx);
  }
    
  //  nBagged = 0;
  for (auto index : idx) {
    //nBagged += (sc[index] == 0 ? 1 : 0);
    sc[index]++;
  }

  return sc;
}


vector<unsigned int> Sampler::binIndices(const vector<unsigned int>& idx) {
  // Sets binPop to respective bin population, then accumulates population
  // of bins to the left.
  // Performance not sensitive to bin width.
  //
  vector<unsigned int> binPop(1 + binIdx(idx.size()));
  for (auto val : idx) {
    binPop[binIdx(val)]++;
  }
  for (unsigned int i = 1; i < binPop.size(); i++) {
    binPop[i] += binPop[i-1];
  }

  // Available index initialzed to one less than total population left of and
  // including bin.  Empty bins have same initial index as bin to the left.
  // This is not a problem, as empty bins are not (re)visited.
  //
  vector<int> idxAvail(binPop.size());
  for (unsigned int i = 0; i < idxAvail.size(); i++) {
    idxAvail[i] = static_cast<int>(binPop[i]) - 1;
  }

  // Writes to the current available index for bin, which is then decremented.
  //
  // Performance degrades if bin width exceeds available cache.
  //
  vector<unsigned int> idxBinned(idx.size());
  for (auto index : idx) {
    int destIdx = idxAvail[binIdx(index)]--;
    idxBinned[destIdx] = index;
  }

  return idxBinned;
}

// Move to Leaf:
void Sampler::consumeTerminals(const PreTree* pretree,
			       const SampleMap& terminalMap) {
  IndexT bagCount = terminalMap.indices.size();
  IndexT extentStart = extentCresc.size();
  IndexT idStart = indexCresc.size();
  IndexT nLeaf = terminalMap.range.size();

  // Pre-grows extent and sample buffers for unordered writes.
  indexCresc.insert(indexCresc.end(), bagCount, 0); // bag-count
  extentCresc.insert(extentCresc.end(), nLeaf, 0);

  // Writes leaf extents for tree, unordered.
  IndexT rangeIdx = 0;
  for (IndexRange range : terminalMap.range) {
    IndexT leafIdx = pretree->getLeafIdx(terminalMap.ptIdx[rangeIdx]);
    extentCresc[extentStart + leafIdx] = range.getExtent();
    rangeIdx++;
  }

  // Accumulates sample index starting positions, in order.
  vector<IndexT> leafStart(nLeaf);
  IndexT startAccum = idStart;
  for (IndexT leafIdx = 0; leafIdx < nLeaf; leafIdx++) {
    leafStart[leafIdx] = exchange(startAccum, startAccum + extentCresc[extentStart + leafIdx]);
  }

  rangeIdx = 0;
  for (IndexRange range : terminalMap.range) {
    IndexT leafIdx = pretree->getLeafIdx(terminalMap.ptIdx[rangeIdx]);
    IndexT idBegin = leafStart[leafIdx];
    for (IndexT idx = range.getStart(); idx != range.getEnd(); idx++) {
      indexCresc[idBegin++] = terminalMap.indices[idx];
    }
    rangeIdx++;
  }
}


size_t Sampler::crescBlockBytes() const {
  if (nuxSamples) {
    return sbCresc.size() * sizeof(SamplerNux);
  }
  else {
    return bagMatrix->getNSlot() * sizeof(BVSlotT);
  }
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


void Sampler::dumpExtent(double extentOut[]) const {
  for (size_t i = 0; i < extentCresc.size(); i++) {
    extentOut[i] = extentCresc[i];
  }
}


void Sampler::dumpIndex(double indexOut[]) const {
  for (size_t i = 0; i < indexCresc.size(); i++) {
    indexOut[i] = indexCresc[i];
  }
}
