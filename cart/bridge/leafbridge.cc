// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leafBridge.cc

   @brief Front-end wrapper for core-level Leaf objects.

   @author Mark Seligman
 */

#include "leafpredict.h"
#include "leafbridge.h"
#include "bagbridge.h"


LeafBridge::LeafBridge(const unsigned int* height,
                             unsigned int nTree,
                             const unsigned char* node,
                             const unsigned int* bagHeight,
                             const unsigned char* bagSample) :
  leaf(make_unique<LeafPredict>(height, nTree, (const Leaf*) node, bagHeight, (const BagSample*) bagSample)) {
}


LeafBridge::~LeafBridge() {
}


void LeafBridge::dump(const BagBridge* bagBridge,
                         vector<vector<size_t> >& rowTree,
                         vector<vector<unsigned int> >& sCountTree,
                         vector<vector<double> >& scoreTree,
                         vector<vector<unsigned int> >& extentTree) const {
  leaf->dump(bagBridge->getBag(), rowTree, sCountTree, scoreTree, extentTree);
}



LeafPredict* LeafBridge::getLeaf() {
  return leaf.get();
}
