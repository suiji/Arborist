// Copyright (C)  2012-2021   Mark Seligman
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
   @file trainR.cc

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#include "trainR.h"
#include "trainbridge.h"
#include "samplerR.h"
#include "forestR.h"
#include "rowSample.h"
#include "rleframeR.h"
#include "rleframe.h"

bool TrainRf::verbose = false;

RcppExport SEXP TrainRF(const SEXP sDeframe, const SEXP sArgList) {
  BEGIN_RCPP

    return TrainRf::train(List(sDeframe), List(sArgList));

  END_RCPP
}


List TrainRf::train(const List& lDeframe, const List& argList) {
  BEGIN_RCPP
  return train(argList, RLEFrameR::unwrap(lDeframe).get());
  END_RCPP
}


List TrainRf::train(const List& argList,
		    const RLEFrame* rleFrame) {
  BEGIN_RCPP

  if (verbose) {
    Rcout << "Beginning training" << endl;
  }
  vector<string> diag;
  unique_ptr<TrainBridge> trainBridge(make_unique<TrainBridge>(rleFrame, as<double>(argList["autoCompress"]), as<bool>(argList["enableCoproc"]), diag));

  initFromArgs(argList, trainBridge.get());

  unique_ptr<TrainRf> trainRF;
  if (as<PredictorT>(argList["nCtg"]) > 0) {
    trainRF = classification(argList, trainBridge.get());
  }
  else {
    trainRF = regression(argList, trainBridge.get());
  }
  List outList = trainRF->summarize(trainBridge.get(), diag);

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
			   TrainBridge* trainBridge) {
  BEGIN_RCPP

  vector<PredictorT> pm = trainBridge->getPredMap();
  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap(pm.begin(), pm.end());

  verbose = as<bool>(argList["verbose"]);
  SamplerR::init(as<bool>(argList["thinLeaves"]));
  
  NumericVector probVecNV((SEXP) argList["probVec"]);
  vector<double> predProb(as<vector<double> >(probVecNV[predMap]));
  trainBridge->initProb(as<PredictorT>(argList["predFixed"]), predProb);

  RowSample::init(as<NumericVector>(argList["rowWeight"]),
                   as<bool>(argList["withRepl"]));
  trainBridge->initSample(as<IndexT>(argList["nSamp"]));

  NumericVector splitQuantNV((SEXP) argList["splitQuant"]);
  vector<double> splitQuant(as<vector<double> >(splitQuantNV[predMap]));
  trainBridge->initSplit(as<unsigned int>(argList["minNode"]),
			 as<unsigned int>(argList["nLevel"]),
			 as<double>(argList["minInfo"]),
			 splitQuant);

  trainBridge->initTree(as<IndexT>(argList["nSamp"]),
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
  SamplerR::deInit();
  trainBridge->deInit();
  END_RCPP
}


unique_ptr<TrainRf> TrainRf::classification(const List& argList,
					    const TrainBridge* trainBridge) {
  IntegerVector yTrain((SEXP) argList["y"]);
  NumericVector classWeight((SEXP) argList["classWeight"]);
  unsigned int nTree = as<unsigned int>(argList["nTree"]);

  IntegerVector yZero = yTrain - 1; // Zero-based translation.
  vector<unsigned int> yzVec(yZero.begin(), yZero.end());
  NumericVector weight = ctgWeight(yZero, classWeight);
  vector<double> weightVec(weight.begin(), weight.end());
  
  unique_ptr<TrainRf> tb = make_unique<TrainRf>(nTree, yTrain);
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += treeChunk) {
    auto chunkThis = treeOff + treeChunk > nTree ? nTree - treeOff : treeChunk;
    auto trainChunk = trainBridge->classification(yzVec,
						  weightVec,
						  CharacterVector(yTrain.attr("levels")).length(),
						  chunkThis,
						  nTree);
    tb->consume(trainChunk.get(), treeOff, chunkThis);
  }

  return tb;
}


NumericVector TrainRf::ctgWeight(const IntegerVector& y,
				 const NumericVector& classWeight) {

  BEGIN_RCPP
    
  auto scaledWeight = clone(classWeight);
  // Default class weight is all unit:  scaling yields 1.0 / nCtg uniformly.
  // All zeroes is a place-holder to indicate balanced scaling:  class weights
  // are proportional to the inverse of the count of the class in the response.
  if (is_true(all(classWeight == 0.0))) {
    NumericVector tb(table(y));
    for (R_len_t i = 0; i < classWeight.length(); i++) {
      scaledWeight[i] = tb[i] == 0.0 ? 0.0 : 1.0 / tb[i];
    }
  }
  NumericVector yWeighted = scaledWeight / sum(scaledWeight); // in [0,1]
  return yWeighted[y];

  END_RCPP
}


void TrainRf::consume(const TrainChunk* train,
                      unsigned int treeOff,
                      unsigned int chunkSize) {
  double scale = safeScale(treeOff + chunkSize);
  sampler->consume(train, treeOff, scale);
  forest->consume(train, treeOff, scale);

  NumericVector infoChunk(train->getPredInfo().begin(), train->getPredInfo().end());
  if (predInfo.length() == 0) {
    predInfo = infoChunk;
  }
  else {
    predInfo = predInfo + infoChunk;
  }
  
  if (verbose) {
    Rcout << treeOff + chunkSize << " trees trained" << endl;
  }
}


List TrainRf::summarize(const TrainBridge* trainBridge,
                        const vector<string>& diag) {
  BEGIN_RCPP
  return List::create(
                      _["predInfo"] = scaleInfo(trainBridge),
                      _["diag"] = diag,
                      _["forest"] = move(forest->wrap()),
		      _["predMap"] = move(trainBridge->getPredMap()),
                      _["sampler"] = move(sampler->wrap())
                      );
  END_RCPP
}


NumericVector TrainRf::scaleInfo(const TrainBridge* trainBridge) {
  BEGIN_RCPP

  vector<PredictorT> pm = trainBridge->getPredMap();
  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap(pm.begin(), pm.end());

  predInfo = predInfo / nTree; // Scales info per-tree.
  return predInfo[predMap]; // Maps back from core order.

  END_RCPP
}


unique_ptr<TrainRf> TrainRf::regression(const List& argList,
					const TrainBridge* trainBridge) {
  NumericVector y(as<NumericVector>(argList["y"]));
  unsigned int nTree = as<unsigned int>(argList["nTree"]);

  vector<double> yVec(y.begin(), y.end());
  unique_ptr<TrainRf> tb = make_unique<TrainRf>(nTree, y);
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += treeChunk) {
    auto chunkThis = treeOff + treeChunk > nTree ? nTree - treeOff : treeChunk;
    auto trainChunk = trainBridge->regression(yVec, chunkThis);
    tb->consume(trainChunk.get(), treeOff, chunkThis);
  }
  return tb;
}


TrainRf::TrainRf(unsigned int nTree_,
                 const NumericVector& yTrain) :
  nTree(nTree_),
  sampler(make_unique<SamplerRegR>(yTrain, nTree)),
  forest(make_unique<FBTrain>(nTree)) {
}


TrainRf::TrainRf(unsigned int nTree_,
                 const IntegerVector& yZero) :
  nTree(nTree_),
  sampler(make_unique<SamplerCtgR>(yZero, nTree)),
  forest(make_unique<FBTrain>(nTree)) {
}



