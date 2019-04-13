// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of ArboristBridgeR.
//
// ArboristBridgeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// ArboristBridgeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file bagBridge.cc

   @brief C++ interface sampled bag.

   @author Mark Seligman
 */

#include "bagBridge.h"
#include "train.h"
#include "bv.h"

BagBridge::BagBridge(unsigned int nRow_, unsigned int nTree_) :
  nRow(nRow_),
  nTree(nTree_),
  rowBytes(BitMatrix::strideBytes(nRow)),
  raw(RawVector(nTree * rowBytes)) {
}


BagBridge::BagBridge(unsigned int nRow_, unsigned int nTree_, const RawVector &raw_) :
  nRow(nRow_),
  nTree(nTree_),
  rowBytes(BitMatrix::strideBytes(nRow)),
  raw(raw_),
  bmRaw(raw.length() > 0 ? make_unique<BitMatrix>((unsigned int*) &raw[0], nTree, nRow) : make_unique<BitMatrix>(0, 0)) {
}


BagBridge::~BagBridge() {
}


void BagBridge::consume(const Train *train, unsigned int treeOff) {
  train->cacheBagRaw((unsigned char*) &raw[treeOff * rowBytes]);
}


const BitMatrix *BagBridge::getRaw() {
  return bmRaw.get();
}


List BagBridge::wrap() {
  BEGIN_RCPP
  return List::create(
                      _["raw"] = move(raw),
                      _["nRow"] = nRow,
                      _["rowBytes"] = rowBytes,
                      _["nTree"] = nTree
                      );

  END_RCPP
}


unique_ptr<BagBridge> BagBridge::unwrap(const List &sTrain, const List &sPredBlock, bool oob) {

  List sBag((SEXP) sTrain["bag"]);
  if (oob) {
    checkOOB(sBag, sPredBlock);
  }

  return make_unique<BagBridge>(as<unsigned int>(sBag["nRow"]),
                                as<unsigned int>(sBag["nTree"]),
                                RawVector((SEXP) sBag["raw"]));
}


SEXP BagBridge::checkOOB(const List& sBag, const List& sPredBlock) {
  BEGIN_RCPP
  if (as<unsigned int>(sBag["nRow"]) == 0)
    stop("Out-of-bag prediction requested but bag empty");

  if (as<unsigned int>(sBag["nRow"]) != as<unsigned int>(sPredBlock["nRow"]))
    stop("Bag and prediction row counts do not agree");

  END_RCPP
}

unique_ptr<BagBridge> BagBridge::unwrap(const List &sTrain) {
  List sBag((SEXP) sTrain["bag"]);
  return make_unique<BagBridge>(as<unsigned int>(sBag["nRow"]),
                                as<unsigned int>(sBag["nTree"]),
                                RawVector((SEXP) sBag["raw"]));
}

