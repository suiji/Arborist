// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictbridge.cc

   @brief Exportable classes and methods from the Predict class.

   @author Mark Seligman
*/

#include "predictbridge.h"
#include "predict.h"
#include "bagbridge.h"
#include "forestbridge.h"
#include "forest.h"
#include "leafbridge.h"
#include "leaf.h"
#include "quant.h"
#include "rleframe.h"
#include "ompthread.h"


PredictBridge::PredictBridge(bool oob,
			     unique_ptr<RLEFrame> rleFrame_,
                             unique_ptr<ForestBridge> forest_,
                             unique_ptr<BagBridge> bag_,
                             unique_ptr<LeafBridge> leaf_,
                             const vector<double>& quantile,
                             unsigned int nThread) :
  rleFrame(move(rleFrame_)),
  bag(move(bag_)),
  forest(move(forest_)),
  leaf(move(leaf_)),
  quant(make_unique<Quant>(static_cast<LeafFrameReg*>(leaf->getLeaf()), bag->getBag(), quantile)),
  predictCore(make_unique<Predict>(bag->getBag(), forest->getForest(), leaf->getLeaf(), rleFrame.get(), quant.get(), oob)) {
  OmpThread::init(nThread);
}


PredictBridge::PredictBridge(bool oob,
			     unique_ptr<RLEFrame> rleFrame_,
                             unique_ptr<ForestBridge> forest_,
                             unique_ptr<BagBridge> bag_,
                             unique_ptr<LeafBridge> leaf_,
                             unsigned int nThread) :
  rleFrame(move(rleFrame_)),
  bag(move(bag_)),
  forest(move(forest_)),
  leaf(move(leaf_)),
  quant(nullptr),
  predictCore(make_unique<Predict>(bag->getBag(), forest->getForest(), leaf->getLeaf(), rleFrame.get(), quant.get(), oob)) {
  OmpThread::init(nThread);
}


PredictBridge::~PredictBridge() {
  OmpThread::deInit();
}


LeafBridge* PredictBridge::getLeaf() const {
  return leaf.get();
}


const vector<double> PredictBridge::getQPred() const {

  return (quant == nullptr || quant->getNRow() == 0) ? vector<double>(0) : quant->getQPred();
}


const vector<double> PredictBridge::getQEst() const {
  return (quant == nullptr || quant->getNRow() == 0) ? vector<double>(0) : quant->getQEst();
}


size_t PredictBridge::getBlockRows(size_t rowCount) {
  return Predict::getBlockRows(rowCount);
}


void PredictBridge::predictBlock(size_t rowStart,
				 size_t extent) const {
  unique_ptr<PredictFrame> frame(make_unique<PredictFrame>(predictCore.get(), extent));
  frame->predictAcross(rowStart);
}
