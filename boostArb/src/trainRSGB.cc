// Copyright (C)  2012-2024   Mark Seligman
//
// This file is part of sgbArb.
//
// sgbArb is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// sgbArb is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file trainRSGB.cc

   @brief C++ interface to R training entry for Rborist package.

   @author Mark Seligman
 */

#include "trainR.h"
#include "trainbridge.h"


SEXP TrainR::initPerInvocation(const List& argList,
			       TrainBridge& trainBridge) {
  BEGIN_RCPP

  vector<unsigned int> pm = trainBridge.getPredMap();
  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap(pm.begin(), pm.end());

  verbose = as<bool>(argList["verbose"]);
  NumericVector probVecNV((SEXP) argList["probVec"]);
  vector<double> predProb(as<vector<double> >(probVecNV[predMap]));
  trainBridge.initProb(as<unsigned int>(argList["predFixed"]), predProb);

  NumericVector splitQuantNV((SEXP) argList["splitQuant"]);
  vector<double> splitQuant(as<vector<double> >(splitQuantNV[predMap]));
  trainBridge.initSplit(as<unsigned int>(argList["minNode"]),
			 as<unsigned int>(argList["nLevel"]),
			 as<double>(argList["minInfo"]),
			 splitQuant);

  trainBridge.initBooster(as<string>(argList["loss"]),
			  as<string>(argList["forestScore"]),
			  as<double>(argList["nu"]),
			  as<bool>(argList["trackFit"]),
			  as<unsigned int>(argList["stopLag"]));
  trainBridge.initNodeScorer(as<string>(argList["nodeScore"]));
  trainBridge.initTree(as<unsigned int>(argList["maxLeaf"]));
  trainBridge.initGrove(as<bool>(argList["thinLeaves"]),
			as<unsigned int>(argList["treeBlock"]));
  trainBridge.initOmp(as<unsigned int>(argList["nThread"]));
  
  if (!Rf_isFactor((SEXP) argList["y"])) {
    NumericVector regMonoNV((SEXP) argList["regMono"]);
    vector<double> regMono(as<vector<double> >(regMonoNV[predMap]));
    trainBridge.initMono(regMono);
  }

  END_RCPP
}


RcppExport SEXP trainSGB(const SEXP sDeframe, const SEXP sSampler, const SEXP sArgList) {
  BEGIN_RCPP

  return TrainR::train(List(sDeframe), List(sSampler), List(sArgList));

  END_RCPP
}
