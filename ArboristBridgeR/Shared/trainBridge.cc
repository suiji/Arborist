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
   @file trainBridge.cc

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#include "trainBridge.h"
#include "rowSample.h"
#include "bagBridge.h"
#include "framemapBridge.h"
#include "rankedsetBridge.h"
#include "forestBridge.h"
#include "leafBridge.h"
#include "leaf.h"
#include "coproc.h"
#include "train.h"

bool TrainBridge::verbose = false;
unsigned int TrainBridge::nCtg = 0;

RcppExport SEXP TrainRF(const SEXP sArgList) {
  BEGIN_RCPP

  List argList(sArgList);
  List predBlock(as<List>(argList["predBlock"]));
  List signature(as<List>(predBlock["signature"]));

  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap((SEXP) signature["predMap"]);
  vector<unsigned int> facCard(as<vector<unsigned int> >(predBlock["facCard"]));

  return TrainBridge::train(argList, predMap, facCard, as<unsigned int>(predBlock["nRow"]));
  END_RCPP
}


List TrainBridge::train(const List &argList,
                        const IntegerVector &predMap,
                        const vector<unsigned int> &facCard,
                        unsigned int nRow) {
  BEGIN_RCPP

  auto frameTrain = FramemapBridge::factoryTrain(facCard, predMap.length(), nRow);
  vector<string> diag;
  auto coproc = Coproc::Factory(as<bool>(argList["enableCoproc"]), diag);
  auto rankedSet = RankedSetBridge::unwrap(argList["rankedSet"],
                                           as<double>(argList["autoCompress"]),
                                           coproc.get(),
                                           frameTrain.get());
  init(argList, frameTrain.get(), predMap);
  List outList;

  if (verbose) {
    Rcout << "Beginning training" << endl;
  }
  
  if (nCtg > 0) {
    outList = classification(IntegerVector((SEXP) argList["y"]),
                             NumericVector((SEXP) argList["classWeight"]),
                             frameTrain.get(),
                             rankedSet->getPair(),
                             predMap,
                             as<unsigned int>(argList["nTree"]),
                             diag);
  }
  else {
    outList = regression(NumericVector((SEXP) argList["y"]),
                         frameTrain.get(),
                         rankedSet->getPair(),
                         predMap,
                         as<unsigned int>(argList["nTree"]),
                         diag);
  }
  if (verbose) {
    Rcout << "Training completed" << endl;
  }

  deInit();
  return move(outList);

  END_RCPP
}


// Employs Rcpp-style temporaries for ease of indexing through
// the predMap[] vector.
SEXP TrainBridge::init(const List &argList,
                       const FrameTrain* frameTrain,
                       const IntegerVector &predMap) {
  BEGIN_RCPP
  verbose = as<bool>(argList["verbose"]);
  LBTrain::init(as<bool>(argList["thinLeaves"]));
  
  NumericVector probVecNV((SEXP) argList["probVec"]);
  vector<double> predProb(as<vector<double> >(probVecNV[predMap]));
  Train::initProb(as<unsigned int>(argList["predFixed"]), predProb);

  NumericVector splitQuantNV((SEXP) argList["splitQuant"]);
  vector<double> splitQuant(as<vector<double> >(splitQuantNV[predMap]));
  Train::initCDF(splitQuant);

  RowSample::init(as<NumericVector>(argList["rowWeight"]),
                   as<bool>(argList["withRepl"]));
  Train::initSample(as<unsigned int>(argList["nSamp"]));
  Train::initSplit(as<unsigned int>(argList["minNode"]),
                   as<unsigned int>(argList["nLevel"]),
                   as<double>(argList["minInfo"]));
  Train::initTree(as<unsigned int>(argList["nSamp"]),
                  as<unsigned int>(argList["minNode"]),
                  as<unsigned int>(argList["maxLeaf"]));
  Train::initBlock(as<unsigned int>(argList["treeBlock"]));
  Train::initOmp(as<unsigned int>(argList["nThread"]));
  
  nCtg = as<unsigned int>(argList["nCtg"]);
  Train::initCtgWidth(nCtg);
  if (nCtg == 0) { // Regression only.
    NumericVector regMonoNV((SEXP) argList["regMono"]);
    vector<double> regMono(as<vector<double> >(regMonoNV[predMap]));
    Train::initMono(frameTrain, regMono);
  }

  END_RCPP
}

SEXP TrainBridge::deInit() {
  BEGIN_RCPP

  nCtg = 0;
  verbose = false;
  LBTrain::deInit();
  Train::deInit();
  END_RCPP
}


NumericVector TrainBridge::ctgProxy(const IntegerVector &y,
                                    const NumericVector &classWeight) {
  BEGIN_RCPP
    
  auto scaledWeight = clone(classWeight);
  if (is_true(all(classWeight == 0.0))) { // Place-holder for balancing.
    NumericVector tb(table(y));
    for (R_len_t i = 0; i < classWeight.length(); i++) {
      scaledWeight[i] = tb[i] == 0.0 ? 0.0 : 1.0 / tb[i];
    }
  }
  scaledWeight = scaledWeight / sum(scaledWeight);

  auto nRow = y.length();
  NumericVector yWeighted = scaledWeight[y];
  RNGScope scope;
  NumericVector rn(runif(nRow));
  return yWeighted + (rn - 0.5) / (2 * nRow * nRow);

  END_RCPP
}


List TrainBridge::classification(const IntegerVector &y,
                                 const NumericVector &classWeight,
                                 const FrameTrain *frameTrain,
                                 const RankedSet *rankedPair,
                                 const IntegerVector &predMap,
                                 unsigned int nTree,
                                 vector<string> &diag) {
  BEGIN_RCPP

  IntegerVector yZero = y - 1; // Zero-based translation.
  auto proxy = ctgProxy(yZero, classWeight);

  unique_ptr<TrainBridge> tb = make_unique<TrainBridge>(nTree, predMap, y);
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += treeChunk) {
    auto chunkThis = treeOff + treeChunk > nTree ? nTree - treeOff : treeChunk;
    auto trainCtg =
      Train::classification(frameTrain,
                            rankedPair,
                            &(as<vector<unsigned int> >(yZero))[0],
                            &proxy[0],
                            classWeight.size(),
                            chunkThis,
                            nTree);
    tb->consume(trainCtg.get(), treeOff, chunkThis);
  }
  return tb->summarize(predMap, diag);

  END_RCPP
}


List TrainBridge::summarize(const IntegerVector &predMap,
                            const vector<string> &diag) {
  BEGIN_RCPP
  return List::create(
                      _["predInfo"] = scalePredInfo(predMap),
                      _["diag"] = diag,
                      _["forest"] = move(forest->wrap()),
                      _["leaf"] = move(leaf->wrap()),
                      _["bag"] = move(bag->wrap())
                      );
  END_RCPP
}


NumericVector TrainBridge::scalePredInfo(const IntegerVector &predMap) {
  BEGIN_RCPP

  predInfo = predInfo / nTree; // Scales info per-tree.
  return predInfo[predMap]; // Maps back from core order.

  END_RCPP
}


List TrainBridge::regression(const NumericVector &y,
                             const FrameTrain *frameTrain,
                             const RankedSet *rankedPair,
                             const IntegerVector &predMap,
                             unsigned int nTree,
                             vector<string> &diag) {
  BEGIN_RCPP
    
  unique_ptr<TrainBridge> tb = make_unique<TrainBridge>(nTree, predMap, y);
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += treeChunk) {
    auto chunkThis = treeOff + treeChunk > nTree ? nTree - treeOff : treeChunk;
    auto trainReg =
      Train::regression(frameTrain,
                        rankedPair,
                        &y[0],
                        chunkThis);
    tb->consume(trainReg.get(), treeOff, chunkThis);
  }
  return tb->summarize(predMap, diag);

  END_RCPP
}


TrainBridge::TrainBridge(unsigned int nTree_,
                         const IntegerVector& predMap,
                         const NumericVector& yTrain) :
  nTree(nTree_),
  bag(make_unique<BagBridge>(yTrain.length(), nTree)),
  forest(make_unique<FBTrain>(nTree)),
  predInfo(NumericVector(predMap.length())),
  leaf(make_unique<LBTrainReg>(yTrain, nTree)) {
  predInfo.fill(0.0);
}


TrainBridge::TrainBridge(unsigned int nTree_,
                         const IntegerVector& predMap,
                         const IntegerVector& yTrain) :
  nTree(nTree_),
  bag(make_unique<BagBridge>(yTrain.length(), nTree)),
  forest(make_unique<FBTrain>(nTree)),
  predInfo(NumericVector(predMap.length())),
  leaf(make_unique<LBTrainCtg>(yTrain, nTree)) {
  predInfo.fill(0.0);
}


void TrainBridge::consume(const Train* train,
                          unsigned int treeOff,
                          unsigned int chunkSize) {
  double scale = safeScale(treeOff + chunkSize);
  bag->consume(train, treeOff);
  forest->consume(train->getForest(), treeOff, scale);
  leaf->consume(train->getLeaf(), treeOff, scale);

  NumericVector infoChunk(train->getPredInfo().begin(), train->getPredInfo().end());
  predInfo = predInfo + infoChunk;

  if (verbose) {
    Rcout << treeOff + chunkSize << " trees trained" << endl;
  }
}

