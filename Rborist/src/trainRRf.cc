// Copyright (C)  2012-2023   Mark Seligman
//
// This file is part of Rborist.
//
// Rborist is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// Rborist is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file trainRRf.cc

   @brief C++ interface to R training entry for Rborist package.

   @author Mark Seligman
 */

#include "trainR.h"
#include "trainbridge.h"


SEXP TrainR::initFromArgs(const List& argList,
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

  trainBridge.initBooster();
  trainBridge.initTree(as<unsigned int>(argList["maxLeaf"]));
  trainBridge.initBlock(as<unsigned int>(argList["treeBlock"]));
  trainBridge.initOmp(as<unsigned int>(argList["nThread"]));
  
  if (!Rf_isFactor((SEXP) argList["y"])) {
    NumericVector regMonoNV((SEXP) argList["regMono"]);
    vector<double> regMono(as<vector<double> >(regMonoNV[predMap]));
    trainBridge.initMono(regMono);
  }

  END_RCPP
}


RcppExport SEXP trainRF(const SEXP sDeframe, const SEXP sSampler, const SEXP sArgList) {
  BEGIN_RCPP

  return TrainR::trainInd(List(sDeframe), List(sSampler), List(sArgList));

  END_RCPP
}
