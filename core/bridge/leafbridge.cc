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

#include "leaf.h"
#include "leafbridge.h"
#include "bagbridge.h"

// For now, cloned implementations:
size_t LeafBridge::getRowPredict() const {
  return getLeaf()->getRowPredict();
}


LeafRegBridge::LeafRegBridge(const unsigned int* height,
                             unsigned int nTree,
                             const unsigned char* node,
                             const unsigned int* bagHeight,
                             const unsigned char* bagSample,
                             const double* yTrain,
                             size_t rowTrain,
                             double trainMean,
                             size_t rowPredict) :
  leaf(make_unique<LeafFrameReg>(height, nTree, (const Leaf*) node, bagHeight, (const BagSample*) bagSample, yTrain, rowTrain, trainMean, rowPredict)) {
}


LeafRegBridge::~LeafRegBridge() {
}


LeafFrame* LeafRegBridge::getLeaf() const {
  return leaf.get();
}


const vector<double>& LeafRegBridge::getYPred() const {
  return leaf->getYPred();
}


void LeafRegBridge::dump(const BagBridge* bagBridge,
                         vector<vector<size_t> >& rowTree,
                         vector<vector<unsigned int> >& sCountTree,
                         vector<vector<double> >& scoreTree,
                         vector<vector<unsigned int> >& extentTree) const {
  leaf->dump(bagBridge->getBag(), rowTree, sCountTree, scoreTree, extentTree);
}


void LeafCtgBridge::dump(const BagBridge* bagBridge,
                         vector<vector<size_t> > &rowTree,
                         vector<vector<unsigned int> > &sCountTree,
                         vector<vector<double> > &scoreTree,
                         vector<vector<unsigned int> > &extentTree,
                         vector<vector<double> > &probTree) const {
  leaf->dump(bagBridge->getBag(), rowTree, sCountTree, scoreTree, extentTree, probTree);
}


LeafCtgBridge::LeafCtgBridge(const unsigned int* height,
                             unsigned int nTree,
                             const unsigned char* node,
                             const unsigned int* bagHeight,
                             const unsigned char* bagSample,
                             const double* weight,
                             unsigned int ctgTrain,
                             size_t rowPredict,
                             bool doProb) :
  leaf(make_unique<LeafFrameCtg>(height, nTree, (const Leaf*) node, bagHeight, (const BagSample*) bagSample, weight, ctgTrain, rowPredict, doProb)) {
}


LeafCtgBridge::~LeafCtgBridge() {
}



LeafFrame* LeafCtgBridge::getLeaf() const {
  return leaf.get();
}


void LeafCtgBridge::vote() {
  leaf->vote();
}

const unsigned int* LeafCtgBridge::getCensus() const {
  return leaf->getCensus();
}

const vector<double>& LeafCtgBridge::getProb() const {
  return leaf->getProb();
}

const vector<unsigned int>& LeafCtgBridge::getYPred() const {
  return leaf->getYPred();
}


unsigned int LeafCtgBridge::getYPred(size_t row) const {
  return leaf->getYPred(row);
}


unsigned int LeafCtgBridge::getCtgTrain() const {
  return leaf->getCtgTrain();
}

unsigned int LeafCtgBridge::ctgIdx(unsigned int ctgTest,
                    unsigned int ctgPred) const {
  return leaf->ctgIdx(ctgTest, ctgPred);
}
