// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of rfR.
//
// rfR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rfR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file bagRf.cc

   @brief C++ interface sampled bag.

   @author Mark Seligman
 */

#include "bagRf.h"
#include "bagbridge.h"
#include "trainbridge.h"


BagRf::BagRf(size_t nObs_, unsigned int nTree_) :
  nObs(nObs_),
  nTree(nTree_),
  rowBytes(BagBridge::strideBytes(nObs)),
  raw(RawVector(nTree * rowBytes)) {
}


BagRf::~BagRf() {
}


void BagRf::consume(const TrainChunk* train, unsigned int treeOff) {
  train->dumpBagRaw((unsigned char*) &raw[treeOff * rowBytes]);
}


List BagRf::wrap() {
  BEGIN_RCPP
  return List::create(
                      _["raw"] = move(raw),
                      _["nRow"] = nObs,
                      _["rowBytes"] = rowBytes,
                      _["nTree"] = nTree
                      );

  END_RCPP
}


unique_ptr<BagBridge> BagRf::unwrap(const List &sTrain, const List &sPredFrame, bool oob) {

  List sBag((SEXP) sTrain["bag"]);
  if (oob) {
    checkOOB(sBag, sPredFrame);
  }

  RawVector raw((SEXP) sBag["raw"]);
  if (raw.length() > 0) {
    return make_unique<BagBridge>(as<unsigned int>(sBag["nTree"]),
                                  as<unsigned int>(sBag["nRow"]),
                                  RawVector((SEXP) sBag["raw"]).begin());
  }
  else {
    return make_unique<BagBridge>();
  }
}


SEXP BagRf::checkOOB(const List& sBag, const List& sPredFrame) {
  BEGIN_RCPP
  if (as<unsigned int>(sBag["nRow"]) == 0)
    stop("Out-of-bag prediction requested with empty bag.");

  if (as<unsigned int>(sBag["nRow"]) != as<unsigned int>(sPredFrame["nRow"]))
    stop("Bag and prediction row counts do not agree.");

  END_RCPP
}


unique_ptr<BagBridge> BagRf::unwrap(const List &sTrain) {
  List sBag((SEXP) sTrain["bag"]);
  return make_unique<BagBridge>(as<unsigned int>(sBag["nTree"]),
                                as<unsigned int>(sBag["nRow"]),
                                (unsigned char*) RawVector((SEXP) sBag["raw"]).begin());
}

