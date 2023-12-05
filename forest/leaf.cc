// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "forest.h"
#include "sampler.h"
#include "samplemap.h"
#include "pretree.h"
#include "response.h"
#include "leaf.h"
#include "ompthread.h"


PackedT RankCount::rankMask = 0;
unsigned int RankCount::rightBits = 0;


unique_ptr<Leaf> Leaf::train(IndexT nObs) {
  RankCount::setMasks(nObs);
  return make_unique<Leaf>();
}


unique_ptr<Leaf> Leaf::predict(const Sampler* sampler,
			       vector<vector<size_t>> extent,
			       vector<vector<vector<size_t>>> index) {
  return make_unique<Leaf>(sampler, std::move(extent), std::move(index));
}


Leaf::Leaf() {
}


Leaf::Leaf(const Sampler* sampler,
	   vector<vector<size_t>> extent_,
	   vector<vector<vector<size_t>>> index_) :
  extent(std::move(extent_)),
  index(std::move(index_)) {
  RankCount::setMasks(sampler->getNObs());
}


Leaf::~Leaf() = default;


Leaf Leaf::unpack(const Sampler* sampler,
		  const double extent_[],
		  const double index_[]) {
  vector<vector<size_t>> extent = unpackExtent(sampler, extent_);
  vector<vector<vector<size_t>>> index = unpackIndex(sampler, extent, index_);
  return Leaf(sampler, std::move(extent), std::move(index));
}


vector<vector<size_t>> Leaf::unpackExtent(const Sampler* sampler,
					  const double extentNum[]) {
  if (extentNum == nullptr) {
    return vector<vector<size_t>>(0);
  }

  unsigned int nTree = sampler->getNRep();
  vector<vector<size_t>> unpacked(nTree);
  size_t idx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    size_t extentTree = 0;
    while (extentTree < sampler->getBagCount(tIdx)) {
      size_t extentLeaf = extentNum[idx++];
      unpacked[tIdx].push_back(extentLeaf);
      extentTree += extentLeaf;
    }
  }
  return unpacked;
}


vector<vector<vector<size_t>>> Leaf::unpackIndex(const Sampler* sampler,
						 const vector<vector<size_t>>& extent,
						 const double numVal[]) {
  if (extent.empty() || numVal == nullptr)
    return vector<vector<vector<size_t>>>(0);

  unsigned int nTree = sampler->getNRep();
  vector<vector<vector<size_t>>> unpacked(nTree);

  size_t idx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    unpacked[tIdx] = vector<vector<size_t>>(extent[tIdx].size());
    for (size_t leafIdx = 0; leafIdx < unpacked[tIdx].size(); leafIdx++) {
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


void Leaf::consumeTerminals(const PreTree* pretree) {
  const SampleMap& terminalMap = pretree->getTerminalMap();
  IndexT bagCount = terminalMap.sampleIndex.size();
  IndexT extentStart = extentCresc.size();
  IndexT idStart = indexCresc.size();
  IndexT nLeaf = terminalMap.range.size();

  // Pre-grows extent and index buffers for unordered writes.
  indexCresc.insert(indexCresc.end(), bagCount, 0);
  extentCresc.insert(extentCresc.end(), nLeaf, 0);

  // Writes leaf extents for tree, unordered.
  IndexT idx = 0;
  for (IndexRange range : terminalMap.range) {
    IndexT leafIdx = pretree->getLeafIdx(terminalMap.ptIdx[idx++]);
    extentCresc[extentStart + leafIdx] = range.getExtent();
  }

  // Accumulates sample index starting positions, in order.
  vector<IndexT> leafStart(nLeaf);
  IndexT startAccum = idStart;
  for (IndexT leafIdx = 0; leafIdx < nLeaf; leafIdx++) {
    leafStart[leafIdx] = exchange(startAccum, startAccum + extentCresc[extentStart + leafIdx]);
  }

#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
  for (OMPBound rangeIdx = 0; rangeIdx < terminalMap.range.size(); rangeIdx++) {
    IndexT leafIdx = pretree->getLeafIdx(terminalMap.ptIdx[rangeIdx]);
    IndexT idBegin = leafStart[leafIdx];
    for (IndexT idx = terminalMap.range[rangeIdx].getStart(); idx != terminalMap.range[rangeIdx].getEnd(); idx++) {
      indexCresc[idBegin++] = terminalMap.sampleIndex[idx];
    }
  }
  }
}



vector<vector<vector<size_t>>> Leaf::countLeafCtg(const Sampler* sampler,
						  const ResponseCtg* response) const {
  unsigned int nTree = sampler->getNRep();
  vector<vector<vector<size_t>>> ctgCount(nTree);
  if (!sampler->hasSamples())
    return ctgCount;
  PredictorT nCtg = response->getNCtg();
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT row = 0;
    vector<PredictorT> sIdx2Ctg(sampler->getBagCount(tIdx));
    for (IndexT sIdx = 0; sIdx != sIdx2Ctg.size(); sIdx++) {
      row += sampler->getDelRow(tIdx, sIdx);
      sIdx2Ctg[sIdx] = response->getCtg(row);
    }
    size_t leafIdx = 0;
    ctgCount[tIdx] = vector<vector<size_t>>(getLeafCount(tIdx));
    for (vector<size_t> sIdxVec : getIndices(tIdx)) {
      ctgCount[tIdx][leafIdx] = vector<size_t>(sIdxVec.size() * nCtg);
      for (size_t sIdx : sIdxVec) {
	PredictorT ctg = sIdx2Ctg[sIdx];
	ctgCount[tIdx][leafIdx][ctg] += sampler->getSCount(tIdx, sIdx);
      }
      leafIdx++;
    }
  }

  return ctgCount;
}


vector<vector<vector<RankCount>>> Leaf::alignRanks(const class Sampler* sampler,
						   const vector<IndexT>& obs2Rank) const {
  unsigned int nTree = sampler->getNRep();
  vector<vector<vector<RankCount>>> rankCount(nTree);
  if (!sampler->hasSamples())
    return rankCount;

  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT obsIdx = 0;
    vector<size_t> sIdx2Rank(sampler->getBagCount(tIdx));
    for (IndexT sIdx = 0 ; sIdx != sIdx2Rank.size(); sIdx++) {
      obsIdx += sampler->getDelRow(tIdx, sIdx);
      sIdx2Rank[sIdx] = obs2Rank[obsIdx];
    }
    size_t leafIdx = 0;
    rankCount[tIdx] = vector<vector<RankCount>>(getLeafCount(tIdx));
    for (vector<size_t> sIdxVec : getIndices(tIdx)) {
      rankCount[tIdx][leafIdx] = vector<RankCount>(sIdxVec.size());
      size_t idx = 0;
      for (size_t sIdx : sIdxVec) {
	rankCount[tIdx][leafIdx][idx++].init(sIdx2Rank[sIdx], sampler->getSCount(tIdx, sIdx));
      }
      leafIdx++;
    }
  }

  return rankCount;
}
