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
   @file samplerR.cc

   @brief C++ interface sampled bag.

   @author Mark Seligman
 */

#include "prng.h"
#include "resizeR.h"
#include "samplerR.h"
#include "samplerbridge.h"
#include "rleframeR.h"

#include <algorithm>


const string SamplerR::strYTrain = "yTrain";
const string SamplerR::strNSamp = "nSamp";
const string SamplerR::strNTree = "nTree"; // EXIT
const string SamplerR::strNRep = "nRep";
const string SamplerR::strSamples = "samples";
const string SamplerR::strHash = "hash";

// [[Rcpp::export]]
RcppExport SEXP rootSample(const SEXP sY,
			   const SEXP sWeight,
			   const SEXP sNSamp,
			   const SEXP sNTree,
			   const SEXP sWithRepl,
			   const SEXP sNHoldout,
			   const SEXP sIdxUndefined) {
  NumericVector weight(as<NumericVector>(sWeight));
  vector<size_t> undefined;
  if (Rf_isInteger(sIdxUndefined)) { // Index type specified by front end.
    IntegerVector undefinedFE(as<NumericVector>(sIdxUndefined));
    undefined = vector<size_t>(undefinedFE.begin(), undefinedFE.end());
  }
  else {
    NumericVector undefinedFE(as<NumericVector>(sIdxUndefined));
    undefined = vector<size_t>(undefinedFE.begin(), undefinedFE.end());
  }

  return SamplerR::rootSample(sY, sNSamp, sNTree, sWithRepl, vector<double>(weight.begin(), weight.end()), sNHoldout, undefined);
}


// [[Rcpp::export]]
List SamplerR::rootSample(const SEXP sY,
			  const SEXP sNSamp,
			  const SEXP sNTree,
			  const SEXP sWithRepl,
			  const vector<double>& weight,
			  const SEXP sNHoldout,
			  const vector<size_t>& undefined) {
  SamplerBridge bridge(as<size_t>(sNSamp), getNObs(sY), as<unsigned int>(sNTree), as<bool>(sWithRepl), weight, as<size_t>(sNHoldout), undefined);
  sampleRepeatedly(bridge);
  return wrap(bridge, sY);
}


size_t SamplerR::countObservations(const List& lSampler) {
  return getNObs(lSampler[strYTrain]);
}



size_t SamplerR::getNObs(const SEXP& sY) {
  return Rf_isFactor(sY) ? as<IntegerVector>(sY).length() : as<NumericVector>(sY).length();
}


void SamplerR::sampleRepeatedly(SamplerBridge& bridge) {
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


// [[Rcpp::export]]
IntegerVector SamplerR::sampleReplace(NumericVector& weight,
				     size_t nSamp) {
  RNGScope scope;
  return IntegerVector(sample(weight.length(), nSamp, true, weight, false));
}


// [[Rcpp::export]]
IntegerVector SamplerR::sampleNoReplace(NumericVector& weight,
					 size_t nSamp) {
  RNGScope scope;
  return IntegerVector(sample(weight.length(), nSamp, false, weight, false));
}


// [[Rcpp::export]]
List SamplerR::wrap(const SamplerBridge& bridge,
		    const SEXP& sY) {
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
}


// [[Rcpp::export]]
List SamplerR::wrap(const SamplerBridge& bridge,
		    const IntegerVector& yTrain) {
  List sampler = List::create(_[strYTrain] = yTrain,
		      _[strSamples] = std::move(bridgeConsume(bridge)),
		      _[strNSamp] = bridge.getNSamp(),
		      _[strNRep] = bridge.getNRep(),
		      _[strNTree] = bridge.getNRep(),
		      _[strHash] = 0
		      );
  sampler.attr("class") = "Sampler";
  return sampler;
}


NumericVector SamplerR::bridgeConsume(const SamplerBridge& bridge) {
  NumericVector blockNum(bridge.getNuxCount());
  bridge.dumpNux(&blockNum[0]);
  return blockNum;
}


// [[Rcpp::export]]
List SamplerR::wrap(const SamplerBridge& bridge,
		    const NumericVector& yTrain) {
  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = std::move(bridgeConsume(bridge)),
			      _[strNSamp] = bridge.getNSamp(),
			      _[strNRep] = bridge.getNRep(),
			      _[strNTree] = bridge.getNRep(),
			      _[strHash] = 0
			);
  sampler.attr("class") = "Sampler";
  return sampler;
}


SamplerBridge SamplerR::unwrapTrain(const List& lSampler) {
  if (Rf_isFactor((SEXP) lSampler[strYTrain])) {
    return makeBridgeTrain(lSampler, as<IntegerVector>(lSampler[strYTrain]));
  }
  else {
    return makeBridgeTrain(lSampler, as<NumericVector>(lSampler[strYTrain]));
  }
}


SamplerBridge SamplerR::makeBridgeTrain(const List& lSampler,
					const IntegerVector& yTrain) {
  return SamplerBridge(coreCtg(yTrain),
		       as<size_t>(lSampler[strNSamp]),
		       as<unsigned int>(lSampler[strNTree]),
		       Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
		       as<CharacterVector>(yTrain.attr("levels")).length());
}


SamplerBridge SamplerR::makeBridgeTrain(const List& lSampler,
					const NumericVector& yTrain) {
  return SamplerBridge(vector<double>(yTrain.begin(), yTrain.end()),
		       as<size_t>(lSampler[strNSamp]),
		       as<unsigned int>(lSampler[strNTree]),
		       Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin());
}


vector<unsigned int> SamplerR::coreCtg(const IntegerVector& yTrain) {
  IntegerVector yZero = yTrain - 1;
  return vector<unsigned int>(yZero.begin(), yZero.end());
}


SamplerBridge SamplerR::unwrapPredict(const List& lSampler,
				      const List& lDeframe,
				      bool bagging) {
  if (bagging)
    checkOOB(lSampler, lDeframe);

  if (Rf_isNumeric((SEXP) lSampler[strYTrain])) {
    return makeBridgeNum(lSampler, lDeframe);
  }
  else if (Rf_isFactor((SEXP) lSampler[strYTrain])) {
    return makeBridgeCtg(lSampler, lDeframe);
  }
  else {
    stop("Unrecognized training response type");
  }
}


// [[Rcpp::export]]
void SamplerR::checkOOB(const List& lSampler, const List& lDeframe) {
  if (Rf_isNull(lSampler[strSamples]))
    stop("Out-of-bag prediction requested with empty sampler.");

  if (getNObs(lSampler[strYTrain]) != as<size_t>((SEXP) lDeframe["nRow"]))
    stop("Bag and prediction row counts do not agree.");
}


SamplerBridge SamplerR::makeBridgeNum(const List& lSampler,
				      const List& lDeframe,
				      bool generic) {
  NumericVector yTrain(as<NumericVector>(lSampler[strYTrain]));
  return SamplerBridge(vector<double>(yTrain.begin(), yTrain.end()),
		       as<size_t>(lSampler[strNSamp]),
		       as<unsigned int>(lSampler[strNTree]),
		       Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
		       generic ? nullptr : RLEFrameR::unwrap(lDeframe));
}


SamplerBridge SamplerR::makeBridgeCtg(const List& lSampler,
				      const List& lDeframe,
				      bool generic) {
  IntegerVector yTrain(as<IntegerVector>(lSampler[strYTrain]));
  return SamplerBridge(coreCtg(yTrain),
		       as<CharacterVector>(yTrain.attr("levels")).length(),
		       as<size_t>(lSampler[strNSamp]),
		       as<unsigned int>(lSampler[strNTree]),
		       Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
		       generic ? nullptr : RLEFrameR::unwrap(lDeframe));
}


SamplerBridge SamplerR::unwrapGeneric(const List& lSampler) {
  List lDummy;
  if (Rf_isNumeric(lSampler[strYTrain]))
    return makeBridgeNum(lSampler, lDummy, true);
  else
    return makeBridgeCtg(lSampler, lDummy, true);
}


unsigned int SamplerR::getNRep(const List& lSampler) {
  return as<unsigned int>(lSampler[strNTree]);
}


SamplerExpand SamplerExpand::unwrap(const List& lSampler) {
  return SamplerExpand(lSampler[SamplerR::strNTree], SamplerR::getNObs(lSampler[SamplerR::strYTrain]));
}
