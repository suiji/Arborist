// Copyright (C)  2012-2018   Mark Seligman
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
#include "rcppSample.h"
#include "train.h"
#include "bagBridge.h"
#include "framemapBridge.h"
#include "rankedsetBridge.h"
#include "forestBridge.h"
#include "leafBridge.h"
#include "leaf.h"
#include "coproc.h"

RcppExport SEXP Train(const SEXP sArgList) {
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

  auto frameTrain = FramemapBridge::FactoryTrain(facCard, predMap.length(), nRow);
  vector<string> diag;
  auto coproc = Coproc::Factory(as<bool>(argList["enableCoproc"]), diag);
  auto rankedSet = RankedSetBridge::unwrap(argList["rankedSet"],
                                           as<double>(argList["autoCompress"]),
                                           coproc.get(),
                                           frameTrain.get());
  init(argList, predMap);
  List outList;

  auto nTree = as<unsigned int>(argList["nTree"]);
  auto bag = make_unique<BagBridge>(nRow, nTree);
  if (as<unsigned int>(argList["nCtg"]) > 0) {
    outList = classification(IntegerVector((SEXP) argList["y"]),
                              NumericVector((SEXP) argList["classWeight"]),
                              frameTrain.get(),
                              rankedSet->GetPair(),
                              predMap,
                              nTree,
                              bag.get(),
                              diag);
  }
  else {
    outList = regression(NumericVector((SEXP) argList["y"]),
                          frameTrain.get(),
                          rankedSet->GetPair(),
                          predMap,
                          nTree,
                          bag.get(),
                          diag);
  }
  Train::DeInit();
  return outList;

  END_RCPP
}


// Employs Rcpp-style temporaries for ease of indexing through
// the predMap[] vector.
SEXP TrainBridge::init(const List &argList, const IntegerVector &predMap) {
  BEGIN_RCPP

  NumericVector probVecNV((SEXP) argList["probVec"]);
  vector<double> predProb(as<vector<double> >(probVecNV[predMap]));
  Train::InitProb(as<unsigned int>(argList["predFixed"]), predProb);

  NumericVector splitQuantNV((SEXP) argList["splitQuant"]);
  vector<double> splitQuant(as<vector<double> >(splitQuantNV[predMap]));
  Train::InitCDF(splitQuant);

  RcppSample::Init(as<NumericVector>(argList["rowWeight"]),
                   as<bool>(argList["withRepl"]));
  Train::InitSample(as<unsigned int>(argList["nSamp"]));
  Train::InitSplit(as<unsigned int>(argList["minNode"]),
                   as<unsigned int>(argList["nLevel"]),
                   as<double>(argList["minInfo"]));
  Train::InitTree(as<unsigned int>(argList["nSamp"]),
                  as<unsigned int>(argList["minNode"]),
                  as<unsigned int>(argList["maxLeaf"]));
  Train::InitLeaf(as<bool>(argList["thinLeaves"]));
  Train::InitBlock(as<unsigned int>(argList["treeBlock"]));

  unsigned int nCtg = as<unsigned int>(argList["nCtg"]);
  Train::InitCtgWidth(nCtg);
  if (nCtg == 0) { // Regression.
    NumericVector regMonoNV((SEXP) argList["regMono"]);
    vector<double> regMono(as<vector<double> >(regMonoNV[predMap]));
    Train::InitMono(regMono);
  }

  END_RCPP
}


// Class weighting constructs a proxy response from category frequency.
// The response is then jittered to diminish the possibility of ties
// during scoring.  The magnitude of the jitter, then, should be scaled
// so that no combination of samples can "vote" themselves into a
// false plurality.
//
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
                                 BagBridge *bag,
                                 vector<string> &diag) {
  BEGIN_RCPP

  IntegerVector yZero = y - 1; // Zero-based translation.
  auto proxy = ctgProxy(yZero, classWeight);

  //  const unsigned int treeChunk = nTree;
  const unsigned int chunkOff = 0;
  //  for (unsigned int chunkOff = 0; chunkOff < nTree; chunkOff += treeChunk) {
  auto trainCtg = Train::Classification(frameTrain,
                           rankedPair,
                           &(as<vector<unsigned int> >(yZero))[0],
                           &proxy[0],
                           classWeight.size(),
                           nTree);
  bag->trainChunk(trainCtg.get(), chunkOff);
  return summarize(trainCtg.get(), bag, predMap, nTree, y, diag);

  END_RCPP
}


List TrainBridge::summarize(const TrainCtg *trainCtg,
                            BagBridge *bag,
                            const IntegerVector &predMap,
                            unsigned int nTree,
                            const IntegerVector &y,
                            const vector<string> &diag) {
  BEGIN_RCPP
  return List::create(
      _["predInfo"] = predInfo(trainCtg->PredInfo(), predMap, nTree),
      _["diag"] = diag,
      _["forest"] = move(ForestBridge::wrap(trainCtg->getForest())),
      _["leaf"] = move(LeafBridge::wrap(trainCtg->getLeaf(),
                                        as<CharacterVector>(y.attr("levels")))),
      _["bag"] = move(bag->wrap())
                      );

  END_RCPP
}


NumericVector TrainBridge::predInfo(const vector<double> &predInfo,
                                          const IntegerVector &predMap,
                                          unsigned int nTree) {
  BEGIN_RCPP

  NumericVector infoOut(predInfo.begin(), predInfo.end());
  infoOut = infoOut / nTree; // Scales info per-tree.
  return infoOut[predMap]; // Maps back from core order.

  END_RCPP
}


List TrainBridge::regression(const NumericVector &y,
                             const FrameTrain *frameTrain,
                             const RankedSet *rankedPair,
                             const IntegerVector &predMap,
                             unsigned int nTree,
                             BagBridge *bag,
                             vector<string> &diag) {
  BEGIN_RCPP
    
  auto yOrdered = clone(y).sort();
  IntegerVector row2Rank = match(y, yOrdered) - 1;

  // Strip mine by smaller chunks:
  const unsigned int treeChunk = nTree;
  const unsigned int chunkOff = 0;
  //  for (unsigned int chunkOff = 0; chunkOff < nTree; chunkOff += treeChunk) {
  auto trainReg = Train::Regression(frameTrain,
                             rankedPair,
                             &y[0],
                             &(as<vector<unsigned int> >(row2Rank))[0],
                                    treeChunk);
  bag->trainChunk(trainReg.get(), chunkOff);
  return summarize(trainReg.get(), bag, predMap, nTree, y, diag);

  END_RCPP
}


List TrainBridge::summarize(const TrainReg *trainReg,
                            BagBridge *bag,
                            const IntegerVector &predMap,
                            unsigned int nTree,
                            const NumericVector &y,
                            const vector<string> &diag) {
  BEGIN_RCPP

  return List::create(
      _["predInfo"] = predInfo(trainReg->PredInfo(), predMap, nTree),
      _["diag"] = diag,
      _["forest"] = move(ForestBridge::wrap(trainReg->getForest())),
      _["leaf"] = move(LeafBridge::wrap(trainReg->getLeaf(), y)),
      _["bag"] = move(bag->wrap())
  );

  END_RCPP
}
