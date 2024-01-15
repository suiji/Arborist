// Copyright (C)  2012-2024   Mark Seligman
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

// [[Rcpp::export]]
void TrainR::initPerInvocation(const List& argList,
			       TrainBridge& trainBridge) {
  vector<unsigned int> pm = trainBridge.getPredMap();
  // Temporary IntegerVector copy for subscripted access.
  IntegerVector predMap(pm.begin(), pm.end());

  verbose = as<bool>(argList[strVerbose]);
  NumericVector probVecNV((SEXP) argList[strProbVec]);
  vector<double> predProb(as<vector<double> >(probVecNV[predMap]));
  trainBridge.initProb(as<unsigned int>(argList[strPredFixed]), predProb);

  NumericVector splitQuantNV((SEXP) argList[strSplitQuant]);
  vector<double> splitQuant(as<vector<double> >(splitQuantNV[predMap]));
  trainBridge.initSplit(as<unsigned int>(argList[strMinNode]),
			 as<unsigned int>(argList[strNLevel]),
			 as<double>(argList[strMinInfo]),
			 splitQuant);

  trainBridge.initBooster(as<string>(argList[strLoss]),
			  as<string>(argList[strForestScore]));
  trainBridge.initNodeScorer(as<string>(argList[strNodeScore]));
  trainBridge.initTree(as<unsigned int>(argList[strMaxLeaf]));
  trainBridge.initSamples(as<vector<double>>(argList[strObsWeight]));
  trainBridge.initGrove(as<bool>(argList[strThinLeaves]),
			as<unsigned int>(argList[strTreeBlock]));
  trainBridge.initOmp(as<unsigned int>(argList[strNThread]));
  
  if (!Rf_isFactor((SEXP) argList[strY])) {
    NumericVector regMonoNV((SEXP) argList[strRegMono]);
    vector<double> regMono(as<vector<double> >(regMonoNV[predMap]));
    trainBridge.initMono(regMono);
  }
  else {
    trainBridge.initCtg(TrainR::ctgWeight(IntegerVector((SEXP) argList[strY]),
					  NumericVector((SEXP) argList[strClassWeight])));
  }
}


// [[Rcpp::export]]
RcppExport SEXP trainRF(const SEXP sDeframe, const SEXP sSampler, const SEXP sArgList) {
  return TrainR::train(List(sDeframe), List(sSampler), List(sArgList));
}
