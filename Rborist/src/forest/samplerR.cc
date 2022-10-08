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

#include "prng.h"
#include "resizeR.h"
#include "samplerbridge.h"
#include "samplerR.h"

#include <algorithm>


const string SamplerR::strYTrain = "yTrain";
const string SamplerR::strNSamp = "nSamp";
const string SamplerR::strNTree = "nTree";
const string SamplerR::strSamples = "samples";


RcppExport SEXP rootSample(const SEXP sY,
			   const SEXP sRowWeight,
			   const SEXP sNSamp,
			   const SEXP sNTree,
			   const SEXP sWithRepl) {
  BEGIN_RCPP

  NumericVector weight;
  if (!Rf_isNull(sRowWeight)) {
    NumericVector rowWeight(as<NumericVector>(sRowWeight));
    weight = rowWeight / sum(rowWeight);
  }
  return SamplerR::rootSample(sY, weight, as<size_t>(sNSamp), as<unsigned int>(sNTree), as<bool>(sWithRepl));

  END_RCPP
}


List SamplerR::rootSample(const SEXP sY,
			  NumericVector& weight, // RCPP method overwrites.
			  size_t nSamp,
			  unsigned int nTree,
			  bool withRepl) {
  size_t nObs = Rf_isFactor(sY) ? as<IntegerVector>(sY).length() : as<NumericVector>(sY).length();
  unique_ptr<SamplerBridge> sb = SamplerBridge::preSample(nSamp, nObs, nTree, withRepl, weight.length() == 0 ? nullptr : &weight[0]);

  // Trees exposed at this level to allow front end to parallelize.
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    sb->sample();
    // Rcpp implementation:
    //  vector<size_t> idx = sampleObs(nSamp, withRepl, weight);
    //  sb->appendSamples(idx);
    //}
  }

  return wrap(sb.get(), sY);
}


vector<size_t> SamplerR::sampleObs(size_t nSamp,
				   bool replace,
				   NumericVector& weight) {
  if (replace) {
    IntegerVector samples = sampleReplace(weight, nSamp);
    vector<size_t> sampleOut(samples.begin(), samples.end());
    return sampleOut;
  }
  else {
    IntegerVector samples = sampleNoReplace(weight, nSamp);
    vector<size_t> sampleOut(samples.begin(), samples.end());
    return sampleOut;
  }
}


IntegerVector SamplerR::sampleReplace(NumericVector& weight,
				     size_t nSamp) {
  BEGIN_RCPP
  RNGScope scope;
  IntegerVector rowSample(sample(weight.length(), nSamp, true, weight, false));
  return rowSample;
  END_RCPP
}


 IntegerVector SamplerR::sampleNoReplace(NumericVector& weight,
					 size_t nSamp) {
  BEGIN_RCPP
  RNGScope scope;
  IntegerVector rowSample(sample(weight.length(), nSamp, false, weight, false));
  return rowSample;
  END_RCPP
}


List SamplerR::wrap(const SamplerBridge* sb,
		    const SEXP sY) {		    
  // Caches the front end's response vector as is.
  if (Rf_isFactor(sY)) {
    return wrap(sb, as<IntegerVector>(sY));
  }
  else {
    return wrap(sb, as<NumericVector>(sY));
  }
}


List SamplerR::wrap(const SamplerBridge* sb,
		    const IntegerVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = std::move(bridgeConsume(sb)),
			      _[strNSamp] = sb->getNSamp(),
			      _[strNTree] = sb->getNTree()
			);
  sampler.attr("class") = "Sampler";

  return sampler;
  END_RCPP
}


NumericVector SamplerR::bridgeConsume(const SamplerBridge* bridge) {
  NumericVector blockNum(bridge->getNuxCount());
  bridge->dumpNux(&blockNum[0]);
  return blockNum;
}


List SamplerR::wrap(const SamplerBridge* sb,
		    const NumericVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = std::move(bridgeConsume(sb)),
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
    return unwrapFac(lSampler, argList);
  }
  else {
    return unwrapNum(lSampler);
  }
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
  return SamplerBridge::readReg(std::move(yTrainCore),
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
					      const List& argList) {
  IntegerVector yTrain((SEXP) lSampler[strYTrain]);
  return SamplerBridge::trainCtg(std::move(coreCtg(yTrain)),
				 as<size_t>(lSampler[strNSamp]),
				 as<unsigned int>(lSampler[strNTree]),
				 Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
				 as<CharacterVector>(yTrain.attr("levels")).length(),
				 std::move(ctgWeight(yTrain, as<NumericVector>(argList["classWeight"]))));
}


vector<double> SamplerR::ctgWeight(const IntegerVector& yTrain,
				   const NumericVector& classWeight) {
  IntegerVector yZero = yTrain - 1;
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
  NumericVector weight = yWeighted[yZero];

  vector<double> weightVec(weight.begin(), weight.end());
  return weightVec;
}


unique_ptr<SamplerBridge> SamplerR::unwrapFac(const List& lSampler,
					      bool bagging) {
  IntegerVector yTrain((SEXP) lSampler[strYTrain]);
  return SamplerBridge::readCtg(std::move(coreCtg(yTrain)),
				as<CharacterVector>(yTrain.attr("levels")).length(),
				as<size_t>(lSampler[strNSamp]),
				as<unsigned int>(lSampler[strNTree]),
				Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
				bagging);
}
