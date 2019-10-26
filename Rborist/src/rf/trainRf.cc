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
   @file trainRf.cc

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#include "trainRf.h"
#include "trainbridge.h"
#include "bagRf.h"
#include "forestRf.h"
#include "leafRf.h"
#include "rowSample.h"
#include "rleframeR.h"

bool TrainRf::verbose = false;

RcppExport SEXP TrainRF(const SEXP sArgList) {
  BEGIN_RCPP

  List argList(sArgList);
  List predFrame(as<List>(argList["predFrame"]));
  List signature(as<List>(predFrame["signature"]));

  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap((SEXP) signature["predMap"]);

  return TrainRf::train(argList, predMap, as<unsigned int>(predFrame["nRow"]));

  END_RCPP
}


List TrainRf::train(const List& argList,
		      const IntegerVector& predMap,
		      unsigned int nRow) {
  SEXP sRLEFrame(argList["summaryRLE"]);
  List rleList(sRLEFrame);
  List rankedFrame = RLEFrameR::checkRankedFrame(rleList["rankedFrame"]);
  List blockNum = RLEFrameR::checkNumRanked((SEXP) rleList["numRanked"]);
  IntegerVector cardFE(Rf_isNull(rleList["cardinality"]) ? IntegerVector(0) : IntegerVector((SEXP) rleList["cardinality"]));

  vector<unsigned int> cardinality(cardFE.begin(), cardFE.end());
  RawVector rleRaw((SEXP) rankedFrame["rle"]);
  NumericVector numVal(Rf_isNull(blockNum["numVal"]) ? NumericVector(0) : NumericVector((SEXP) blockNum["numVal"]));
  IntegerVector numOff(Rf_isNull(blockNum["numOff"]) ? IntegerVector(0) : IntegerVector((SEXP) blockNum["numOff"]));
  unique_ptr<RLEFrame> rleFrame(make_unique<RLEFrame>(nRow,
						      cardinality,
						      rleRaw.length() / RLECresc::unitSize(),
						      (const RLEVal<unsigned int>*) &rleRaw[0],
						      (unsigned int) numOff.size(),
						      (const double*) &numVal[0],
						      (const unsigned int*) &numOff[0]));
  return train(argList, predMap, rleFrame.get());
}


List TrainRf::train(const List& argList,
                    const IntegerVector& predMap,
		    const RLEFrame* rleFrame) {
  BEGIN_RCPP

  if (verbose) {
    Rcout << "Beginning training" << endl;
  }
  vector<string> diag;
  unique_ptr<TrainBridge> trainBridge(make_unique<TrainBridge>(rleFrame, as<double>(argList["autoCompress"]), as<bool>(argList["enableCoproc"]), diag));
  initFromArgs(argList, trainBridge.get(), predMap);

  List outList;
  if (as<unsigned int>(argList["nCtg"]) > 0) {
    outList = classification(argList,
                             trainBridge.get(),
                             predMap,
			     diag);
  }
  else {
    outList = regression(argList,
			 trainBridge.get(),
                         predMap,
			 diag);
  }
  if (verbose) {
    Rcout << "Training completed" << endl;
  }

  deInit(trainBridge.get());
  return outList;

  END_RCPP
}


// Employs Rcpp-style temporaries for ease of indexing through
// the predMap[] vector.
SEXP TrainRf::initFromArgs(const List& argList,
                   TrainBridge* trainBridge,
                   const IntegerVector& predMap) {
  BEGIN_RCPP
  verbose = as<bool>(argList["verbose"]);
  LBTrain::init(as<bool>(argList["thinLeaves"]));
  
  NumericVector probVecNV((SEXP) argList["probVec"]);
  vector<double> predProb(as<vector<double> >(probVecNV[predMap]));
  trainBridge->initProb(as<unsigned int>(argList["predFixed"]), predProb);

  RowSample::init(as<NumericVector>(argList["rowWeight"]),
                   as<bool>(argList["withRepl"]));
  trainBridge->initSample(as<unsigned int>(argList["nSamp"]));

  NumericVector splitQuantNV((SEXP) argList["splitQuant"]);
  vector<double> splitQuant(as<vector<double> >(splitQuantNV[predMap]));
  trainBridge->initSplit(as<unsigned int>(argList["minNode"]),
                   as<unsigned int>(argList["nLevel"]),
			 as<double>(argList["minInfo"]),
			 splitQuant);

  trainBridge->initTree(as<unsigned int>(argList["nSamp"]),
                  as<unsigned int>(argList["minNode"]),
                  as<unsigned int>(argList["maxLeaf"]));
  trainBridge->initBlock(as<unsigned int>(argList["treeBlock"]));
  trainBridge->initOmp(as<unsigned int>(argList["nThread"]));
  
  unsigned int nCtg = as<unsigned int>(argList["nCtg"]);
  trainBridge->initCtgWidth(nCtg);
  if (nCtg == 0) { // Regression only.
    NumericVector regMonoNV((SEXP) argList["regMono"]);
    vector<double> regMono(as<vector<double> >(regMonoNV[predMap]));
    trainBridge->initMono(regMono);
  }

  END_RCPP
}

SEXP TrainRf::deInit(TrainBridge* trainBridge) {
  BEGIN_RCPP

  verbose = false;
  LBTrain::deInit();
  trainBridge->deInit();
  END_RCPP
}


NumericVector TrainRf::ctgProxy(const IntegerVector &y,
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


List TrainRf::classification(const List& argList,
                             const TrainBridge* trainBridge,
                             const IntegerVector &predMap,
                             vector<string> &diag) {
  BEGIN_RCPP

  IntegerVector y((SEXP) argList["y"]);
  NumericVector classWeight((SEXP) argList["classWeight"]);
  unsigned int nTree = as<unsigned int>(argList["nTree"]);

  IntegerVector yZero = y - 1; // Zero-based translation.
  vector<unsigned int> yzVec(yZero.begin(), yZero.end());
  auto proxy = ctgProxy(yZero, classWeight);

  unique_ptr<TrainRf> tb = make_unique<TrainRf>(nTree, predMap, y);
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += treeChunk) {
    auto chunkThis = treeOff + treeChunk > nTree ? nTree - treeOff : treeChunk;
    auto trainChunk = trainBridge->classification(&yzVec[0],
                                                &proxy[0],
                                                classWeight.size(),
                                                chunkThis,
                                                nTree);
    tb->consume(trainChunk.get(), treeOff, chunkThis);
  }
  return tb->summarize(predMap, diag);

  END_RCPP
}


void TrainRf::consume(const TrainChunk* train,
                      unsigned int treeOff,
                      unsigned int chunkSize) {
  bag->consume(train, treeOff);

  double scale = safeScale(treeOff + chunkSize);
  forest->consume(train, treeOff, scale);
  leaf->consume(train, treeOff, scale);

  NumericVector infoChunk(train->getPredInfo().begin(), train->getPredInfo().end());
  predInfo = predInfo + infoChunk;

  if (verbose) {
    Rcout << treeOff + chunkSize << " trees trained" << endl;
  }
}


List TrainRf::summarize(const IntegerVector& predMap,
                        const vector<string>& diag) {
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


NumericVector TrainRf::scalePredInfo(const IntegerVector& predMap) {
  BEGIN_RCPP

  predInfo = predInfo / nTree; // Scales info per-tree.
  return predInfo[predMap]; // Maps back from core order.

  END_RCPP
}


List TrainRf::regression(const List& argList,
                         const TrainBridge* trainBridge,
                         const IntegerVector& predMap,
                         vector<string>& diag) {
  BEGIN_RCPP

  NumericVector y((SEXP) argList["y"]);
  unsigned int nTree = as<unsigned int>(argList["nTree"]);
  
  unique_ptr<TrainRf> tb = make_unique<TrainRf>(nTree, predMap, y);
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += treeChunk) {
    auto chunkThis = treeOff + treeChunk > nTree ? nTree - treeOff : treeChunk;
    auto trainChunk = trainBridge->regression(&y[0], chunkThis);
    tb->consume(trainChunk.get(), treeOff, chunkThis);
  }
  return tb->summarize(predMap, diag);

  END_RCPP
}


TrainRf::TrainRf(unsigned int nTree_,
                 const IntegerVector& predMap,
                 const NumericVector& yTrain) :
  nTree(nTree_),
  bag(make_unique<BagRf>(yTrain.length(), nTree)),
  forest(make_unique<FBTrain>(nTree)),
  predInfo(NumericVector(predMap.length())),
  leaf(make_unique<LBTrainReg>(yTrain, nTree)) {
  predInfo.fill(0.0);
}


TrainRf::TrainRf(unsigned int nTree_,
                 const IntegerVector& predMap,
                 const IntegerVector& yTrain) :
  nTree(nTree_),
  bag(make_unique<BagRf>(yTrain.length(), nTree)),
  forest(make_unique<FBTrain>(nTree)),
  predInfo(NumericVector(predMap.length())),
  leaf(make_unique<LBTrainCtg>(yTrain, nTree)) {
  predInfo.fill(0.0);
}



