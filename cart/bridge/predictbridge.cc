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


PredictBridge::PredictBridge(unique_ptr<RLEFrame> rleFrame_,
                             unique_ptr<ForestBridge> forestBridge_,
                             unique_ptr<BagBridge> bagBridge_,
                             unique_ptr<LeafBridge> leafBridge_,
			     bool importance_,
                             const vector<double>& quantile,
                             unsigned int nThread) :
  rleFrame(move(rleFrame_)),
  bagBridge(move(bagBridge_)),
  forestBridge(move(forestBridge_)),
  leafBridge(move(leafBridge_)),
  importance(importance_),
  quant(make_unique<Quant>(static_cast<LeafFrameReg*>(leafBridge->getLeaf()), bagBridge->getBag(), quantile)) {
  OmpThread::init(nThread);
}


PredictBridge::PredictBridge(unique_ptr<RLEFrame> rleFrame_,
                             unique_ptr<ForestBridge> forestBridge_,
                             unique_ptr<BagBridge> bagBridge_,
                             unique_ptr<LeafBridge> leafBridge_,
			     bool importance_,
                             unsigned int nThread) :
  rleFrame(move(rleFrame_)),
  bagBridge(move(bagBridge_)),
  forestBridge(move(forestBridge_)),
  leafBridge(move(leafBridge_)),
  importance(importance_),
  quant(nullptr) {
  OmpThread::init(nThread);
}


PredictBridge::~PredictBridge() {
  OmpThread::deInit();
}


void PredictBridge::predict() const {
  unique_ptr<Predict> predictCore  = make_unique<Predict>(bagBridge->getBag(), forestBridge->getForest(), leafBridge->getLeaf(), rleFrame.get(), quant.get());
  predictCore->predict(importance);
}


LeafBridge* PredictBridge::getLeaf() const {
  return leafBridge.get();
}


const vector<double> PredictBridge::getQPred() const {

  return (quant == nullptr || quant->getNRow() == 0) ? vector<double>(0) : quant->getQPred();
}


const vector<double> PredictBridge::getQEst() const {
  return (quant == nullptr || quant->getNRow() == 0) ? vector<double>(0) : quant->getQEst();
}
