// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forestbridge.cc

   @brief Front-end wrapper for core-level Forest objects.

   @author Mark Seligman
 */

#include "forest.h"
#include "forestbridge.h"
#include "decnoderw.h"
#include "forestrw.h"
#include "typeparam.h"
#include "bv.h"

using namespace std;

ForestBridge::ForestBridge(unsigned int treeChunk) :
  forest(make_unique<Forest>(treeChunk)) {
}


ForestBridge::ForestBridge(unsigned int nTree,
			   const double nodeExtent[],
			   const complex<double> treeNode[],
			   const double score[],
			   const double facExtent[],
                           const unsigned char facSplit[],
			   const unsigned char facObserved[]) :
  forest(make_unique<Forest>(DecNodeRW::unpackNodes(treeNode, nodeExtent, nTree),
			     DecNodeRW::unpackScores(score, nodeExtent, nTree),
			     DecNodeRW::unpackBits(facSplit, facExtent, nTree),
			     DecNodeRW::unpackBits(facObserved, facExtent, nTree))) {
}


ForestBridge::ForestBridge(ForestBridge&& fb) :
  forest(std::exchange(fb.forest, nullptr)) {
}


ForestBridge::~ForestBridge() = default;


void ForestBridge::init(unsigned int nPred) {
  Forest::init(nPred);
}


void ForestBridge::deInit() {
  Forest::deInit();
}


unsigned int ForestBridge::getNTree() const {
  return forest->getNTree();
}


const vector<size_t>& ForestBridge::getNodeExtents() const {
  return forest->getNodeExtents();
}


const vector<size_t>& ForestBridge::getFacExtents() const {
  return forest->getFacExtents();
}


Forest* ForestBridge::getForest() const {
  return forest.get();
}


size_t ForestBridge::getFactorBytes() const {
  return forest->getFactorBytes();
}


void ForestBridge::dumpTree(complex<double> treeOut[]) const {
  forest->cacheNode(treeOut);
}


size_t ForestBridge::getNodeCount() const {
  return forest->getNodeCount();
}


void ForestBridge::dumpScore(double scoreOut[]) const {
  forest->cacheScore(scoreOut);
}


void ForestBridge::dumpFactorRaw(unsigned char facOut[]) const {
  forest->cacheFacRaw(facOut);
}


void ForestBridge::dumpFactorObserved(unsigned char obsOut[]) const {
  forest->cacheObservedRaw(obsOut);
}


void ForestBridge::dump(vector<vector<unsigned int> >& predTree,
                        vector<vector<double> >& splitTree,
                        vector<vector<size_t> >& lhDelTree,
                        vector<vector<unsigned char> >& facSplitTree,
			vector<vector<double>>& scoreTree) const {
  ForestRW::dump(forest.get(), predTree, splitTree, lhDelTree, facSplitTree, scoreTree);
}

    
