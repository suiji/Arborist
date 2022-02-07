// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "sampler.h"
#include "samplemap.h"
#include "pretree.h"
#include "response.h"
#include "leaf.h"


PackedT RankCount::rankMask = 0;
unsigned int RankCount::rightBits = 0;


unique_ptr<Leaf> Leaf::train(IndexT nObs,
			     bool thin) {
  RankCount::setMasks(nObs);
  return make_unique<Leaf>(thin);
}


unique_ptr<Leaf> Leaf::predict(const Sampler* sampler,
			       bool thin,
			       vector<vector<size_t>> extent,
			       vector<vector<vector<size_t>>> index) {
  RankCount::setMasks(sampler->getNObs());
  return make_unique<Leaf>(sampler, thin, extent, index);
}


Leaf::Leaf(bool thin_)
  : thin(thin_) {
}


Leaf::Leaf(const Sampler* sampler,
	   bool thin_,
	   vector<vector<size_t>> extent_,
	   vector<vector<vector<size_t>>> index_) :
  thin(thin_),
  extent(extent_),
  index(index_) {
}


Leaf::~Leaf() {
  RankCount::unsetMasks();
}


void Leaf::consumeTerminals(const PreTree* pretree,
			    const SampleMap& terminalMap) {
  if (thin)
    return;
  
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



vector<vector<vector<size_t>>> Leaf::countLeafCtg(const Sampler* sampler,
						  const ResponseCtg* response) const {
  unsigned int nTree = sampler->getNTree();
  vector<vector<vector<size_t>>> ctgCount(nTree);
  if (!sampler->hasSamples())
    return ctgCount;
  PredictorT nCtg = response->getNCtg();
  //size_t treeIdx = 0; // Absolute sample index.
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT row = 0;
    vector<PredictorT> sIdx2Ctg(sampler->getBagCount(tIdx));
    for (IndexT sIdx = 0; sIdx != sIdx2Ctg.size(); sIdx++) {
      row += sampler->getDelRow(tIdx, sIdx);//treeIdx + sIdx);
      sIdx2Ctg[sIdx] = response->getCtg(row);
    }
    size_t leafIdx = 0;
    ctgCount[tIdx] = vector<vector<size_t>>(getLeafCount(tIdx));
    for (vector<size_t> sIdxVec : getIndices(tIdx)) {
      ctgCount[tIdx][leafIdx] = vector<size_t>(sIdxVec.size() * nCtg);
      for (size_t sIdx : sIdxVec) {
	PredictorT ctg = sIdx2Ctg[sIdx];
	ctgCount[tIdx][leafIdx][ctg] += sampler->getSCount(tIdx, sIdx);//treeIdx + sIdx);
      }
      leafIdx++;
    }
    //treeIdx += sampler->getBagCount(tIdx);
  }

  return ctgCount;
}


vector<vector<vector<RankCount>>> Leaf::alignRanks(const class Sampler* sampler,
						   const vector<IndexT>& row2Rank) const {
  unsigned int nTree = sampler->getNTree();
  vector<vector<vector<RankCount>>> rankCount(nTree);
  if (!sampler->hasSamples())
    return rankCount;

  //  size_t treeIdx = 0;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT row = 0;
    vector<size_t> sIdx2Rank(sampler->getBagCount(tIdx));
    for (IndexT sIdx = 0 ; sIdx != sIdx2Rank.size(); sIdx++) {
      row += sampler->getDelRow(tIdx, sIdx);//treeIdx + sIdx);
      sIdx2Rank[sIdx] = row2Rank[row];
    }
    size_t leafIdx = 0;
    rankCount[tIdx] = vector<vector<RankCount>>(getLeafCount(tIdx));
    for (vector<size_t> sIdxVec : getIndices(tIdx)) {
      rankCount[tIdx][leafIdx] = vector<RankCount>(sIdxVec.size());
      size_t idx = 0;
      for (size_t sIdx : sIdxVec) {
	rankCount[tIdx][leafIdx][idx++].init(sIdx2Rank[sIdx], sampler->getSCount(tIdx, sIdx));//treeIdx + sIdx));
      }
      leafIdx++;
    }
    //    treeIdx += sampler->getBagCount(tIdx);
  }

  return rankCount;
}
