// Copyright (C)  2012-2022   Mark Seligman
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

#include "forestbridge.h"
#include "samplerbridge.h"
#include "leafbridge.h"
#include "trainR.h"
#include "trainbridge.h"
#include "samplerR.h"
#include "leafR.h"
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

  List outList;
  if (as<unsigned int>(argList["nCtg"]) > 0) {
    outList = classification(argList, trainBridge.get(), diag);
  }
  else {
    outList = regression(argList, trainBridge.get(), diag);
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
			   TrainBridge* trainBridge) {
  BEGIN_RCPP

  vector<unsigned int> pm = trainBridge->getPredMap();
  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap(pm.begin(), pm.end());

  verbose = as<bool>(argList["verbose"]);
  NumericVector probVecNV((SEXP) argList["probVec"]);
  vector<double> predProb(as<vector<double> >(probVecNV[predMap]));
  trainBridge->initProb(as<unsigned int>(argList["predFixed"]), predProb);

  RowSample::init(as<NumericVector>(argList["rowWeight"]),
                   as<bool>(argList["withRepl"]));

  NumericVector splitQuantNV((SEXP) argList["splitQuant"]);
  vector<double> splitQuant(as<vector<double> >(splitQuantNV[predMap]));
  trainBridge->initSplit(as<unsigned int>(argList["minNode"]),
			 as<unsigned int>(argList["nLevel"]),
			 as<double>(argList["minInfo"]),
			 splitQuant);

  trainBridge->initTree(as<unsigned int>(argList["maxLeaf"]));
  trainBridge->initBlock(as<unsigned int>(argList["treeBlock"]));
  trainBridge->initOmp(as<unsigned int>(argList["nThread"]));
  
  unsigned int nCtg = as<unsigned int>(argList["nCtg"]);
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
  trainBridge->deInit();
  END_RCPP
}


List TrainRf::classification(const List& argList,
			     const TrainBridge* trainBridge,
			     vector<string>& diag) {
  IntegerVector yTrain((SEXP) argList["y"]);
  NumericVector classWeight((SEXP) argList["classWeight"]);
  unsigned int nTree = as<unsigned int>(argList["nTree"]);

  IntegerVector yZero = yTrain - 1; // Zero-based translation.
  vector<unsigned int> yzVec(yZero.begin(), yZero.end());
  NumericVector weight = ctgWeight(yZero, classWeight);
  vector<double> weightVec(weight.begin(), weight.end());

  unsigned int nSamp = as<unsigned int>(argList["nSamp"]);
  bool thinLeaves = as<bool>(argList["thinLeaves"]);
  
  unique_ptr<TrainRf> trainRf = make_unique<TrainRf>(nSamp, nTree);
  unsigned int nCtg = CharacterVector(yTrain.attr("levels")).length();
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += treeChunk) {
    auto chunkThis = treeOff + treeChunk > nTree ? nTree - treeOff : treeChunk;
    unique_ptr<SamplerBridge> sb = SamplerBridge::crescCtg(yzVec, nSamp, chunkThis, nCtg, weightVec);
    ForestBridge fb(chunkThis);
    unique_ptr<LeafBridge> lb = LeafBridge::FactoryTrain(yzVec.size(), thinLeaves);
    auto trainedChunk = trainBridge->train(fb, sb.get(), lb.get());
    trainRf->consume(fb, sb.get(), lb.get(), treeOff, chunkThis);
    trainRf->consumeInfo(trainedChunk.get());
  }

  return trainRf->summarize(trainBridge, yTrain, diag);
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


void TrainRf::consume(const ForestBridge& fb,
		      const SamplerBridge* sb,
		      const LeafBridge* lb,
                      unsigned int treeOff,
                      unsigned int chunkSize) const {
  double scale = safeScale(treeOff + chunkSize);
  sampler->bridgeConsume(sb, scale);
  forest->bridgeConsume(fb, treeOff, scale);
  leaf->bridgeConsume(lb, scale);
  
  if (verbose) {
    Rcout << treeOff + chunkSize << " trees trained" << endl;
  }
}


void TrainRf::consumeInfo(const TrainedChunk* train) {
  NumericVector infoChunk(train->getPredInfo().begin(), train->getPredInfo().end());
  if (predInfo.length() == 0) {
    predInfo = infoChunk;
  }
  else {
    predInfo = predInfo + infoChunk;
  }
}


List TrainRf::summarize(const TrainBridge* trainBridge,
			const IntegerVector& yTrain,
			const vector<string>& diag) {
  BEGIN_RCPP
  return List::create(
                      _["predInfo"] = scaleInfo(trainBridge),
                      _["diag"] = diag,
                      _["forest"] = move(forest->wrap()),
		      _["predMap"] = move(trainBridge->getPredMap()),
                      _["sampler"] = move(sampler->wrap(yTrain)),
		      _["leaf"] = move(leaf->wrap())
                      );
  END_RCPP
}


List TrainRf::summarize(const TrainBridge* trainBridge,
			const NumericVector& yTrain,
			const vector<string>& diag) {
  BEGIN_RCPP

  return List::create(
                      _["predInfo"] = scaleInfo(trainBridge),
                      _["diag"] = diag,
                      _["forest"] = move(forest->wrap()),
		      _["predMap"] = move(trainBridge->getPredMap()),
                      _["sampler"] = move(sampler->wrap(yTrain)),
		      _["leaf"] = move(leaf->wrap())
                      );
  END_RCPP
}


NumericVector TrainRf::scaleInfo(const TrainBridge* trainBridge) {
  BEGIN_RCPP

  vector<unsigned int> pm = trainBridge->getPredMap();
  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap(pm.begin(), pm.end());

  predInfo = predInfo / nTree; // Scales info per-tree.
  return predInfo[predMap]; // Maps back from core order.

  END_RCPP
}


List TrainRf::regression(const List& argList,
			 const TrainBridge* trainBridge,
			 vector<string>& diag) {
  NumericVector yTrain(as<NumericVector>(argList["y"]));
  unsigned int nTree = as<unsigned int>(argList["nTree"]);
  unsigned int nSamp = as<unsigned int>(argList["nSamp"]);
  bool thinLeaves = as<bool>(argList["thinLeaves"]);

  vector<double> yVec(yTrain.begin(), yTrain.end());
  unique_ptr<TrainRf> trainRf = make_unique<TrainRf>(nSamp, nTree);
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += treeChunk) {
    auto chunkThis = treeOff + treeChunk > nTree ? nTree - treeOff : treeChunk;
    unique_ptr<SamplerBridge> sb = SamplerBridge::crescReg(yVec, nSamp, chunkThis);
    ForestBridge fb(chunkThis);
    unique_ptr<LeafBridge> lb = LeafBridge::FactoryTrain(yVec.size(), thinLeaves);
    auto trainedChunk = trainBridge->train(fb, sb.get(), lb.get());
    trainRf->consume(fb, sb.get(), lb.get(), treeOff, chunkThis);
    trainRf->consumeInfo(trainedChunk.get());
  }
  return trainRf->summarize(trainBridge, yTrain, diag);
}


TrainRf::TrainRf(unsigned int nSamp_,
		 unsigned int nTree_) :
  nSamp(nSamp_),
  nTree(nTree_),
  sampler(make_unique<SamplerR>(nSamp, nTree)),
  leaf(make_unique<LeafR>()),
  forest(make_unique<FBTrain>(nTree)) {
}
