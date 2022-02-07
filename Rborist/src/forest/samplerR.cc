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

#include "resizeR.h"
#include "samplerR.h"
#include "samplerbridge.h"

const string SamplerR::strYTrain = "yTrain";
const string SamplerR::strNSamp = "nSamp";
const string SamplerR::strNTree = "nTree";
const string SamplerR::strSamples = "samples";


SamplerR::SamplerR(unsigned int nSamp_,
		   unsigned int nTree_) :
  nSamp(nSamp_),
  nTree(nTree_),
  blockNum(NumericVector(0)),
  nuxTop(0) {
}


void SamplerR::bridgeConsume(const SamplerBridge* bridge,
			     double scale) {
  size_t nuxCount = bridge->getNuxCount();
  if (nuxTop + nuxCount > static_cast<size_t>(blockNum.length())) {
    blockNum = move(ResizeR::resizeNum(blockNum, nuxTop, nuxCount, scale));
  }
  bridge->dumpNux(&blockNum[nuxTop]);
  nuxTop += nuxCount;
}


List SamplerR::wrap(const IntegerVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = move(blockNum),
			      _[strNSamp] = nSamp,
			      _[strNTree] = nTree
			);
  sampler.attr("class") = "Sampler";

  return sampler;
  END_RCPP
}


List SamplerR::wrap(const NumericVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = move(blockNum),
			      _[strNSamp] = nSamp,
			      _[strNTree] = nTree
			);
  sampler.attr("class") = "Sampler";
  return sampler;
  END_RCPP
}


unique_ptr<SamplerBridge> SamplerR::unwrap(const List& lTrain,
					   const List& lDeframe,
					   bool bagging) {
  List lSampler((SEXP) lTrain["sampler"]);
  if (bagging)
    checkOOB(lSampler, as<size_t>((SEXP) lDeframe["nRow"]));
  return unwrap(lSampler, bagging);
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


unique_ptr<SamplerBridge> SamplerR::unwrap(const List& lSampler,
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
				as<unsigned int>(lSampler[strNSamp]),
				as<unsigned int>(lSampler[strNTree]),
				Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
				bagging);
}


unique_ptr<SamplerBridge> SamplerR::unwrapFac(const List& lSampler,
					      bool bagging) {
    IntegerVector yTrain((SEXP) lSampler[strYTrain]);
    IntegerVector yZero = yTrain - 1;
    vector<unsigned int> yTrainCore(yZero.begin(), yZero.end());
    return SamplerBridge::readCtg(move(yTrainCore),
				  as<CharacterVector>(yTrain.attr("levels")).length(),
				  as<unsigned int>(lSampler[strNSamp]),
				  as<unsigned int>(lSampler[strNTree]),
				  Rf_isNull(lSampler[strSamples]) ? nullptr : NumericVector((SEXP) lSampler[strSamples]).begin(),
				  bagging);
}
