// Copyright (C)  2012-2024   Mark Seligman
//
// This file is part of RboristBase.
//
// RboristBase is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RboristBase is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RboristBase.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file trainR.cc

   @brief C++ interface to R entry for training.

   @author Mark Seligman
 */

#include "grovebridge.h"
#include "leafbridge.h"
#include "trainR.h"
#include "trainbridge.h"
#include "rleframeR.h"
#include "rleframe.h"
#include "samplerR.h"
#include "signatureR.h"


bool TrainR::verbose = false;

const string TrainR::strY = "y";
const string TrainR::strVersion = "version";
const string TrainR::strSignature = "signature";
const string TrainR::strSamplerHash = "samplerHash";
const string TrainR::strPredInfo = "predInfo";
const string TrainR::strPredMap = "predMap";
const string TrainR::strForest = "forest";
const string TrainR::strLeaf = "leaf";
const string TrainR::strDiagnostic = "diag";
const string TrainR::strClassName = "arbTrain";
const string TrainR::strAutoCompress = "autoCompress";
const string TrainR::strEnableCoproc = "enableCoproc";
const string TrainR::strVerbose = "verbose";
const string TrainR::strProbVec = "probVec";
const string TrainR::strPredFixed = "predFixed";
const string TrainR::strSplitQuant ="splitQuant";
const string TrainR::strMinNode = "minNode";
const string TrainR::strNLevel = "nLevel";
const string TrainR::strMinInfo = "minInfo";
const string TrainR::strLoss = "loss";
const string TrainR::strForestScore = "forestScore";
const string TrainR::strNodeScore = "nodeScore";
const string TrainR::strMaxLeaf = "maxLeaf";
const string TrainR::strObsWeight ="obsWeight";
const string TrainR::strThinLeaves =  "thinLeaves";
const string TrainR::strTreeBlock = "treeBlock";
const string TrainR::strNThread = "nThread";
const string TrainR::strRegMono = "regMono";
const string TrainR::strClassWeight = "classWeight";


// [[Rcpp::export]]
List TrainR::train(const List& lDeframe, const List& lSampler, const List& argList) {
  if (verbose) {
    Rcout << "Beginning training" << endl;
  }

  vector<string> diag;
  TrainBridge trainBridge(RLEFrameR::unwrap(lDeframe), as<double>(argList[strAutoCompress]), as<bool>(argList[strEnableCoproc]), diag);
  initPerInvocation(argList, trainBridge);

  TrainR trainR(lSampler);
  trainR.trainGrove(trainBridge);
  List outList = trainR.summarize(trainBridge, lDeframe, lSampler, argList, diag);

  if (verbose) {
    Rcout << "Training completed" << endl;
  }

  deInit();
  return outList;
}


TrainR::TrainR(const List& lSampler) :
  samplerBridge(SamplerR::unwrapTrain(lSampler)),
  nTree(samplerBridge.getNRep()),
  leaf(LeafR()),
  forest(FBTrain(nTree)) {
}


vector<double> TrainR::ctgWeight(const IntegerVector& yTrain,
				 const NumericVector& classWeight) {
  // All-zeroes is a place-holder denoting balanced weighting:  a
  // sampled class's weight is proportional to the inverse of its
  // population in the response.
  if (is_true(all(classWeight == 0.0))) {
    vector<double> tabledWeight;
    NumericVector tb(table(IntegerVector(yTrain - 1)));
    for (double wt : classWeight) {
      tabledWeight.push_back(wt == 0.0 ? 0.0 : 1.0 / wt);
    }
    return tabledWeight;
  }
  else {
    return vector<double>(classWeight.begin(), classWeight.end());
  }
}


void TrainR::deInit() {
  verbose = false;
  TrainBridge::deInit();
}


// [[Rcpp::export]]
List TrainR::summarize(const TrainBridge& trainBridge,
		       const List& lDeframe,
		       const List& lSampler,
		       const List& argList,
		       const vector<string>& diag) {
  List trainArb = List::create(
			       _[strVersion] = as<String>(argList[strVersion]),
			       _[strSignature] = lDeframe[strSignature],
			       _[strSamplerHash] = lSampler[SamplerR::strHash],
			       _[strPredInfo] = scaleInfo(trainBridge),
			       _[strPredMap] = std::move(trainBridge.getPredMap()),
			       _[strForest] = std::move(forest.wrap()),
			       _[strLeaf] = std::move(leaf.wrap()),
			       _[strDiagnostic] = diag
                      );
  trainArb.attr("class") = strClassName;
  return trainArb;
}


// [[Rcpp::export]]
NumericVector TrainR::scaleInfo(const TrainBridge& trainBridge) const {
  vector<unsigned int> pm = trainBridge.getPredMap();
  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap(pm.begin(), pm.end());

  // Mapbs back from core order and scales info per-tree.
  return as<NumericVector>(predInfo[predMap]) / nTree;
}


void TrainR::trainGrove(const TrainBridge& trainBridge) {
  for (unsigned int treeOff = 0; treeOff < nTree; treeOff += groveSize) {
    auto chunkThis = treeOff + groveSize > nTree ? nTree - treeOff : groveSize;
    LeafBridge lb(samplerBridge);
    unique_ptr<GroveBridge> gb = GroveBridge::train(trainBridge, samplerBridge, treeOff, chunkThis, lb);
    consume(gb.get(), lb, treeOff, chunkThis);
  }
  forest.scoreDescConsume(trainBridge);
}


void TrainR::consume(const GroveBridge* grove,
		     const LeafBridge& lb,
		     unsigned int treeOff,
		     unsigned int chunkSize) {
  double scale = safeScale(treeOff + chunkSize);
  forest.groveConsume(grove, treeOff, scale);
  leaf.bridgeConsume(lb, scale);

  NumericVector infoGrove(grove->getPredInfo().begin(), grove->getPredInfo().end());
  if (predInfo.length() == 0) {
    predInfo = infoGrove;
  }
  else {
    predInfo = predInfo + infoGrove;
  }  
  if (verbose) {
    Rcout << treeOff + chunkSize << " trees trained" << endl;
  }
}


// [[Rcpp::export]]
RcppExport SEXP expandTrainRcpp(SEXP sTrain) {
  return TrainR::expand(List(sTrain));
}


// [[Rcpp::export]]
List TrainR::expand(const List& lTrain) {
  IntegerVector predMap(as<IntegerVector>(lTrain[strPredMap]));
  TrainBridge::init(predMap.length());
  List level = SignatureR::getLevel(lTrain);
  List ffe =
    List::create(_[strPredMap] = IntegerVector(predMap),
                 _["factorMap"] = IntegerVector(predMap.end() - level.length(), predMap.end()),
                 _["predLevel"] = level,
		 _["predFactor"] = SignatureR::getFactor(lTrain),
                 _["forest"] = ForestExpand::expand(lTrain, predMap)
                 );

  TrainBridge::deInit();
  ffe.attr("class") = "expandTrain";
  return ffe;
}
