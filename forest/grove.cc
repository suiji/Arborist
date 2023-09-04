// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file grove.cc

   @brief Main entry from front end for training.

   @author Mark Seligman
*/

#include "bv.h"
#include "grove.h"
#include "predictorframe.h"
#include "frontier.h"
#include "pretree.h"
#include "leaf.h"
#include "sampler.h"
#include "nodescorer.h"

#include <algorithm>

unsigned int Grove::trainBlock = 0;


void Grove::initBlock(unsigned int trainBlock_) {
  trainBlock = trainBlock_;
}


void Grove::deInit() {
  trainBlock = 0;
}


Grove::Grove(const PredictorFrame* frame,
	     const IndexRange& range) :
  forestRange(range),
  nodeScorer(NodeScorer::makeScorer()),
  predInfo(vector<double>(frame->getNPred())),
  nodeCresc(make_unique<NodeCresc>()),
  fbCresc(make_unique<FBCresc>()) {
}


void Grove::train(const PredictorFrame* frame,
		  const Sampler * sampler,
		  Leaf* leaf) {
  for (unsigned treeStart = forestRange.getStart(); treeStart < forestRange.getEnd(); treeStart += trainBlock) {
    auto treeBlock = blockProduce(frame, sampler, treeStart, min(treeStart + trainBlock, static_cast<unsigned int>(forestRange.getEnd())));
    blockConsume(treeBlock, leaf);
  }
  splitUpdate(frame);
}


void FBCresc::appendBits(const BV& splitBits_,
			 const BV& observedBits_,
			 size_t bitEnd) {
  size_t nSlot = splitBits_.appendSlots(splitBits, bitEnd);
  (void) observedBits_.appendSlots(observedBits, bitEnd);
  extents.push_back(nSlot);
}


vector<unique_ptr<PreTree>> Grove::blockProduce(const PredictorFrame* frame,
						const Sampler* sampler,
						unsigned int treeStart,
						unsigned int treeEnd) {
  vector<unique_ptr<PreTree>> block;
  for (unsigned int tIdx = treeStart; tIdx < treeEnd; tIdx++) {
    block.emplace_back(Frontier::oneTree(frame, this, sampler, tIdx));
  }

  return block;
}

 
void Grove::blockConsume(const vector<unique_ptr<PreTree>>& treeBlock,
			 Leaf* leaf) {
  for (auto & pretree : treeBlock) {
    pretree->consume(this, leaf);
  }
}


void Grove::consumeInfo(const vector<double>& info) {
  for (IndexT predIdx = 0; predIdx < predInfo.size(); predIdx++) {
    predInfo[predIdx] += info[predIdx];
  }
}


size_t Grove::getNodeCount() const {
  return scoresCresc.size();
}


void Grove::cacheNode(complex<double> complexOut[]) const {
  nodeCresc->dump(complexOut);
}


void Grove::cacheScore(double scoreOut[]) const {
  for (size_t i = 0; i != scoresCresc.size(); i++)
    scoreOut[i] = scoresCresc[i];
}


const vector<size_t>& Grove::getFacExtents() const {
  return fbCresc->getExtents();
}


size_t Grove::getFactorBytes() const {
  return fbCresc->getFactorBytes();
}


void Grove::cacheFacRaw(unsigned char rawOut[]) const {
  fbCresc->dumpSplitBits(rawOut);
}


void Grove::cacheObservedRaw(unsigned char observedOut[]) const {
  fbCresc->dumpObserved(observedOut);
}
  

