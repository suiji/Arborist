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

ForestBridge::ForestBridge(const unsigned int* height,
                           size_t nTree,
                           const unsigned char* node,
                           unsigned int* facSplit,
                           const unsigned int* facHeight) :
  forest(make_unique<Forest>(height, nTree, (const CartNode*) node, facSplit, facHeight)) {
}


ForestBridge::~ForestBridge() {
}


size_t ForestBridge::nodeSize() {
  return sizeof(CartNode);
}


unsigned int ForestBridge::getNTree() const {
  return forest->getNTree();
}


Forest* ForestBridge::getForest() const {
  return forest.get();
}

void ForestBridge::dump(vector<vector<unsigned int> >& predTree,
                        vector<vector<double> >& splitTree,
                        vector<vector<unsigned int> >& lhDelTree,
                        vector<vector<unsigned int> >& facSplitTree) const {
  forest->dump(predTree, splitTree, lhDelTree, facSplitTree);
}

    
