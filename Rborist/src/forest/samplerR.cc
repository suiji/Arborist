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
   @file samplerR.cc

   @brief C++ interface sampled bag.

   @author Mark Seligman
 */

#include "rowsample.h"
#include "resizeR.h"
#include "samplerR.h"
#include "samplerbridge.h"

const string SamplerR::strYTrain = "yTrain";
const string SamplerR::strNSamp = "nSamp";
const string SamplerR::strNTree = "nTree";
const string SamplerR::strSamples = "samples";


RcppExport SEXP rootSample(const SEXP sDeframe, const SEXP sArgList) {
  BEGIN_RCPP

  return SamplerR::sample(List(sDeframe), List(sArgList));

  END_RCPP
}


List SamplerR::sample(const List& lDeframe,
		      const List& argList) {
  RowSample::init(as<NumericVector>(argList["rowWeight"]),
		  as<bool>(argList["withRepl"]));
  unsigned int nTree = as<unsigned int>(argList["nTree"]);
  unique_ptr<SamplerBridge> sb = SamplerBridge::preSample(as<unsigned int>(argList["nSamp"]),
							  as<size_t>(lDeframe["nRow"]),
							  nTree);
  sb->sample(nTree); // Produces samples

  // Caches the front end's response vector as is.
  if (Rf_isFactor((SEXP) argList["y"])) {
    return wrap(sb.get(), as<IntegerVector>((SEXP) argList["y"]));
  }
  else {
    return wrap(sb.get(), as<NumericVector>((SEXP) argList["y"]));
  }
}


NumericVector SamplerR::bridgeConsume(const SamplerBridge* bridge) {
  NumericVector blockNum(bridge->getNuxCount());
  bridge->dumpNux(&blockNum[0]);
  return blockNum;
}


List SamplerR::wrap(const SamplerBridge* sb,
		    const IntegerVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = move(bridgeConsume(sb)),
			      _[strNSamp] = sb->getNSamp(),
			      _[strNTree] = sb->getNTree()
			);
  sampler.attr("class") = "Sampler";

  return sampler;
  END_RCPP
}


List SamplerR::wrap(const SamplerBridge* sb,
		    const NumericVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = move(bridgeConsume(sb)),
			      _[strNSamp] = sb->getNSamp(),
			      _[strNTree] = sb->getNTree()
			);
  sampler.attr("class") = "Sampler";
  return sampler;
  END_RCPP
}


unique_ptr<SamplerBridge> SamplerR::unwrapTrain(const List& lSampler,
						const List& argList) {
  if (Rf_isFactor((SEXP) lSampler[strYTrain])) {
    return unwrapFac(lSampler, move(weightVec(lSampler, argList)));
  }
  else {
    return unwrapNum(lSampler);
  }
}


vector<double> SamplerR::weightVec(const List& lSampler, const List& argList) {
  NumericVector classWeight((SEXP) argList["classWeight"]);
  NumericVector weight = ctgWeight(as<IntegerVector>(lSampler[strYTrain]) - 1, classWeight);
  vector<double> weightVec(weight.begin(), weight.end());
  return weightVec;
}


NumericVector SamplerR::ctgWeight(const IntegerVector& yZero,
				  const NumericVector& classWeight) {

  BEGIN_RCPP
    
  auto scaledWeight = clone(classWeight);
  // Default class weight is all unit:  scaling yields 1.0 / nCtg uniformly.
  // All zeroes is a place-holder to indicate balanced scaling:  class weights
  // are proportional to the inverse of the count of the class in the response.
  if (is_true(all(classWeight == 0.0))) {
    NumericVector tb(table(yZero));
    for (R_len_t i = 0; i < classWeight.length(); i++) {
      scaledWeight[i] = tb[i] == 0.0 ? 0.0 : 1.0 / tb[i];
    }
  }
  NumericVector yWeighted = scaledWeight / sum(scaledWeight); // in [0,1]
  return yWeighted[yZero];

  END_RCPP
}


unique_ptr<SamplerBridge> SamplerR::unwrapPredict(const List& lSampler,
						  const List& lDeframe,
						  bool bagging) {
  if (bagging)
    checkOOB(lSampler, as<size_t>((SEXP) lDeframe["nRow"]));
  return unwrapPredict(lSampler, bagging);
}


SEXP SamplerR::checkOOB(const List& lSampler, size_t nRow) {
  BEGIN_RCPP
  if (Rf_isNull(lSampler[strSamples]))
    stop("Out-of-bag prediction requested with empty sampler.");

  R_xlen_t nObs = Rf_isNumeric((SEXP) lSampler[strYTrain]) ? as<NumericVector>(lSampler[strYTrain]).length() : as<IntegerVector>(lSampler[strYTrain]).length();

  if (static_cast<size_t>(nObs) != nRow)
    stop("Bag and prediction row counts do not agree.");

  END_RCPP
}


unique_ptr<SamplerBridge> SamplerR::unwrapPredict(const List& lSampler,
						  bool bagging) {
  if (Rf_isNumeric((SEXP) lSampler[strYTrain])) {
    return unwrapNum(lSampler, bagging);
  }
  else if (Rf_isFactor((SEXP) lSampler[strYTrain])) {
    return unwrapFac(lSampler, bagging);
  }
  else {
    stop("Unrecognized training response type");
  }
}


unique_ptr<SamplerBridge> SamplerR::unwrapNum(const List& lSampler,
					      bool bagging) {
  NumericVector yTrain((SEXP) lSampler[strYTrain]);
  vector<double> yTrainCore(yTrain.begin(), yTrain.end());
  return SamplerBridge::readReg(move(yTrainCore),
				as<size_t>(lSampler[strNSamp]),
				as<unsigned int>(lSampler[strNTree]),
				Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
				bagging);
}


vector<unsigned int> SamplerR::coreCtg(const IntegerVector& yTrain) {
  IntegerVector yZero = yTrain - 1;
  vector<unsigned int> yTrainCore(yZero.begin(), yZero.end());
  return yTrainCore;
}


unique_ptr<SamplerBridge> SamplerR::unwrapFac(const List& lSampler,
					      vector<double> weights) {
  IntegerVector yTrain((SEXP) lSampler[strYTrain]);
  return SamplerBridge::trainCtg(move(coreCtg(yTrain)),
				 as<size_t>(lSampler[strNSamp]),
				 as<unsigned int>(lSampler[strNTree]),
				 Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
				 as<CharacterVector>(yTrain.attr("levels")).length(),
				 weights);
}


unique_ptr<SamplerBridge> SamplerR::unwrapFac(const List& lSampler,
					      bool bagging) {
  IntegerVector yTrain((SEXP) lSampler[strYTrain]);
  return SamplerBridge::readCtg(move(coreCtg(yTrain)),
				as<CharacterVector>(yTrain.attr("levels")).length(),
				as<size_t>(lSampler[strNSamp]),
				as<unsigned int>(lSampler[strNTree]),
				Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
				bagging);
}
