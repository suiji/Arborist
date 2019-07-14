// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bagbridge.cc

   @brief Front-end wrapper for core-level Bag objects.

   @author Mark Seligman
 */

#include "bagbridge.h"
#include "bag.h"

#include <memory>
using namespace std;

BagBridge::BagBridge(size_t nObs,
                     unsigned int nTree,
                     unsigned char* raw) :
  bag(make_unique<Bag>((unsigned int*) raw, nObs, nTree)) {
}


BagBridge::BagBridge() : bag(make_unique<Bag>()) {
}


BagBridge::~BagBridge() {
}


size_t BagBridge::strideBytes(size_t nObs) {
  return BitMatrix::strideBytes(nObs);
}


const Bag* BagBridge::getBag() const {
  return bag.get();
}


unsigned int BagBridge::getNObs() const {
  return bag->getNObs();
}


unsigned int BagBridge::getNTree() const {
  return bag->getNTree();
}
