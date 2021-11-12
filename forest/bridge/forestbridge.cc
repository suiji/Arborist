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
			   const unsigned char* node,
                           unsigned char* facSplit) :
  forest(make_unique<Forest>(nTree,
			     reinterpret_cast<const CartNode*>(node),
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


Forest* ForestBridge::getForest() const {
  return forest.get();
}


size_t ForestBridge::getFactorBytes() const {
  return forest->getFactorBytes();
}


void ForestBridge::dumpTreeRaw(unsigned char treeOut[]) const {
  forest->cacheNodeRaw(treeOut);
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

    
