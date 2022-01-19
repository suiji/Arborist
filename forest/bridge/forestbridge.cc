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
#include "cartnode.h"

#include <memory>
using namespace std;


ForestBridge::ForestBridge(unsigned int treeChunk) :
  forest(make_unique<Forest>(treeChunk)) {
}


ForestBridge::ForestBridge(unsigned int nTree,
			   const double* nodeExtent,
			   const unsigned char* node,
			   const double* scores,
			   const double* facExtent,
                           unsigned char* facSplit) :
  forest(make_unique<Forest>(nTree,
			     nodeExtent,
			     reinterpret_cast<const CartNode*>(node),
			     scores,
			     facExtent,
			     reinterpret_cast<unsigned int*>(facSplit))) {
}


ForestBridge::~ForestBridge() {
}


size_t ForestBridge::getNodeBytes() const {
  return forest->getNodeBytes();
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


void ForestBridge::dumpTreeRaw(unsigned char treeOut[]) const {
  forest->cacheNodeRaw(treeOut);
}


size_t ForestBridge::getScoreSize() const {
  return forest->getScoreSize();
}


void ForestBridge::dumpScore(double scoreOut[]) const {
  forest->cacheScore(scoreOut);
}


void ForestBridge::dumpFactorRaw(unsigned char facOut[]) const {
  forest->cacheFacRaw(facOut);
}


void ForestBridge::dump(vector<vector<unsigned int> >& predTree,
                        vector<vector<double> >& splitTree,
                        vector<vector<unsigned int> >& lhDelTree,
                        vector<vector<unsigned int> >& facSplitTree) const {
  forest->dump(predTree, splitTree, lhDelTree, facSplitTree);
}

    
