// Copyright (C)  2012-2023   Mark Seligman
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
   @file samplerR.cc

   @brief C++ interface sampled bag.

   @author Mark Seligman
 */

#include "prng.h"
#include "resizeR.h"
#include "samplerR.h"
#include "samplerbridge.h"

#include <algorithm>


const string SamplerR::strYTrain = "yTrain";
const string SamplerR::strNSamp = "nSamp";
const string SamplerR::strNTree = "nTree";
const string SamplerR::strSamples = "samples";
const string SamplerR::strHash = "hash";


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

  return SamplerR::rootSample(sY, sNSamp, sNTree, sWithRepl, weight);

  END_RCPP
}


List SamplerR::rootSample(const SEXP sY,
			  const SEXP sNSamp,
			  const SEXP sNTree,
			  const SEXP sWithRepl,
			  const NumericVector& weight) {
  BEGIN_RCPP

  SamplerBridge bridge(as<size_t>(sNSamp), getNObs(sY), as<unsigned int>(sNTree), as<bool>(sWithRepl), weight.length() == 0 ? nullptr : &weight[0]);
  sampleTrees(bridge);
  return wrap(bridge, sY);

  END_RCPP
}


size_t SamplerR::countObservations(const List& lSampler) {
  return getNObs(lSampler[strYTrain]);
}



size_t SamplerR::getNObs(const SEXP& sY) {
  return Rf_isFactor(sY) ? as<IntegerVector>(sY).length() : as<NumericVector>(sY).length();
}


void SamplerR::sampleTrees(SamplerBridge& bridge) {
  // May be parallelized if a thread-safe PRNG is available.
  for (unsigned int tIdx = 0; tIdx < bridge.getNRep(); tIdx++) {
    bridge.sample();
  }
}


vector<size_t> SamplerR::sampleObs(size_t nSamp,
				   bool replace,
				   NumericVector& weight) {
  IntegerVector samples = replace ? sampleReplace(weight, nSamp) : sampleNoReplace(weight, nSamp);
  return vector<size_t>(samples.begin(), samples.end());
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


List SamplerR::wrap(const SamplerBridge& bridge,
		    const SEXP& sY) {
  BEGIN_RCPP

  List sampler;
  // Caches the front end's response vector as is.
  if (Rf_isFactor(sY)) {
    sampler = wrap(bridge, as<IntegerVector>(sY));
  }
  else {
    sampler = wrap(bridge, as<NumericVector>(sY));
  }

  Environment digestEnv = Environment::namespace_env("digest");
  Function digestFun = digestEnv["digest"];
  sampler[strHash] = digestFun(sampler, "md5");
  sampler.attr("class") = "Sampler";

  return sampler;

  END_RCPP
}


List SamplerR::wrap(const SamplerBridge& bridge,
		    const IntegerVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = std::move(bridgeConsume(bridge)),
			      _[strNSamp] = bridge.getNSamp(),
			      _[strNTree] = bridge.getNRep(),
			      _[strHash] = 0
			);

  return sampler;
  END_RCPP
}


NumericVector SamplerR::bridgeConsume(const SamplerBridge& bridge) {
  NumericVector blockNum(bridge.getNuxCount());
  bridge.dumpNux(&blockNum[0]);
  return blockNum;
}


List SamplerR::wrap(const SamplerBridge& bridge,
		    const NumericVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = std::move(bridgeConsume(bridge)),
			      _[strNSamp] = bridge.getNSamp(),
			      _[strNTree] = bridge.getNRep(),
			      _[strHash] = 0
			);
  sampler.attr("class") = "Sampler";
  return sampler;

  END_RCPP
}


SamplerBridge SamplerR::unwrapTrain(const List& lSampler,
				    const List& argList) {
  if (Rf_isFactor((SEXP) lSampler[strYTrain])) {
    return makeBridgeTrain(lSampler, as<IntegerVector>(lSampler[strYTrain]), argList);
  }
  else {
    return makeBridgeTrain(lSampler, as<NumericVector>(lSampler[strYTrain]));
  }
}


SamplerBridge SamplerR::makeBridgeTrain(const List& lSampler,
					const IntegerVector& yTrain,
					const List& argList) {
  return SamplerBridge(std::move(coreCtg(yTrain)),
		       as<size_t>(lSampler[strNSamp]),
		       as<unsigned int>(lSampler[strNTree]),
		       Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
		       as<CharacterVector>(yTrain.attr("levels")).length(),
		       std::move(ctgWeight(yTrain, as<NumericVector>(argList["classWeight"]))));
}


SamplerBridge SamplerR::makeBridgeTrain(const List& lSampler,
					const NumericVector& yTrain) {
  return SamplerBridge(std::move(vector<double>(yTrain.begin(), yTrain.end())),
		       as<size_t>(lSampler[strNSamp]),
		       as<unsigned int>(lSampler[strNTree]),
		       Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin());
}


vector<unsigned int> SamplerR::coreCtg(const IntegerVector& yTrain) {
  IntegerVector yZero = yTrain - 1;
  vector<unsigned int> yTrainCore(yZero.begin(), yZero.end());
  return yTrainCore;
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


SamplerBridge SamplerR::unwrapPredict(const List& lSampler,
				      const List& lDeframe,
				      const List& lArgs) {
  bool bagging = as<bool>(lArgs["bagging"]);
  if (bagging)
    checkOOB(lSampler, lDeframe);

  if (Rf_isNumeric((SEXP) lSampler[strYTrain])) {
    return makeBridgeNum(lSampler, bagging);
  }
  else if (Rf_isFactor((SEXP) lSampler[strYTrain])) {
    return makeBridgeCtg(lSampler, bagging);
  }
  else {
    stop("Unrecognized training response type");
  }
}


SEXP SamplerR::checkOOB(const List& lSampler, const List& lDeframe) {
  BEGIN_RCPP
  if (Rf_isNull(lSampler[strSamples]))
    stop("Out-of-bag prediction requested with empty sampler.");

  if (getNObs(lSampler[strYTrain]) != as<size_t>((SEXP) lDeframe["nRow"]))
    stop("Bag and prediction row counts do not agree.");

  END_RCPP
}


SamplerBridge SamplerR::makeBridgeNum(const List& lSampler,
				      bool bagging) {
  NumericVector yTrain(as<NumericVector>(lSampler[strYTrain]));
  return SamplerBridge(std::move(vector<double>(yTrain.begin(), yTrain.end())),
		       as<size_t>(lSampler[strNSamp]),
		       as<unsigned int>(lSampler[strNTree]),
		       Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
		       bagging);
}


SamplerBridge SamplerR::makeBridgeCtg(const List& lSampler,
				   bool bagging) {
  IntegerVector yTrain(as<IntegerVector>(lSampler[strYTrain]));
  return SamplerBridge(std::move(coreCtg(yTrain)),
		       as<CharacterVector>(yTrain.attr("levels")).length(),
		       as<size_t>(lSampler[strNSamp]),
		       as<unsigned int>(lSampler[strNTree]),
		       Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
		       bagging);
}


SamplerBridge SamplerR::unwrapGeneric(const List& lSampler) {
  return SamplerBridge(getNObs(lSampler[strYTrain]),
		       Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
		       as<size_t>(lSampler[strNSamp]),
		       as<unsigned int>(lSampler[strNTree]));
}


unsigned int SamplerR::getNRep(const List& lSampler) {
  return as<unsigned int>(lSampler[strNTree]);
}


SamplerExpand SamplerExpand::unwrap(const List& lSampler) {
  return SamplerExpand(lSampler[SamplerR::strNTree], SamplerR::getNObs(lSampler[SamplerR::strYTrain]));
}
