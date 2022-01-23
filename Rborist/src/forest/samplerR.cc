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
const string SamplerR::strExtent = "extent";
const string SamplerR::strIndex = "index";


SamplerR::SamplerR(unsigned int nSamp_,
		   unsigned int nTree_,
		   bool nux_) :
  nSamp(nSamp_),
  nTree(nTree_),
  nux(nux_),
  rawTop(0),
  extentTop(0),
  indexTop(0) {
}


void SamplerR::bridgeConsume(const SamplerBridge* bridge,
			     double scale) {
  size_t blockBytes = bridge->getBlockBytes(); // # sample bytes in chunk.
  if (rawTop + blockBytes > static_cast<size_t>(blockRaw.length())) {
    blockRaw = move(ResizeR::resizeRaw(blockRaw, rawTop, blockBytes, scale));
  }
  bridge->dumpRaw(&blockRaw[rawTop]);
  rawTop += blockBytes;

  size_t extentSize = bridge->getExtentSize();
  if (extentTop + extentSize > static_cast<size_t>(extent.length())) {
    extent = move(ResizeR::resizeNum(extent, extentTop, extentSize, scale));
  }
  bridge->dumpExtent(&extent[extentTop]);
  extentTop += extentSize;

  size_t indexSize = bridge->getIndexSize();
  if (indexTop + indexSize > static_cast<size_t>(index.length())) {
    index = move(ResizeR::resizeNum(index, indexTop, indexSize, scale));
  }
  bridge->dumpIndex(&index[indexTop]);
  indexTop += indexSize;
}


List SamplerR::wrap(const IntegerVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = move(blockRaw),
			      _[strNSamp] = nSamp,
			      _[strNTree] = nTree,
			      _[strExtent] = move(extent),
			      _[strIndex] = move(index)
			);
  sampler.attr("class") = "Sampler";
  as<RawVector>(sampler[strSamples]).attr("class") = nux ? "nux" : "bits";
  
  return sampler;
  END_RCPP
}


List SamplerR::wrap(const NumericVector& yTrain) {
  BEGIN_RCPP

  List sampler = List::create(_[strYTrain] = yTrain,
			      _[strSamples] = move(blockRaw),
			      _[strNSamp] = nSamp,
			      _[strNTree] = nTree,
			      _[strExtent] = move(extent),
			      _[strIndex] = move(index)
			);
  sampler.attr("class") = "Sampler";
  as<RawVector>(sampler[strSamples]).attr("class") = nux ? "nux" : "bits";
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
  return make_unique<SamplerBridge>(move(yTrainCore),
				    as<IndexT>(lSampler[strNSamp]),
				    as<unsigned int>(lSampler[strNTree]),
				    as<string>(as<RawVector>(lSampler[strSamples]).attr("class")) == "nux",  
				    Rf_isNull(lSampler[strSamples]) ? nullptr : reinterpret_cast<unsigned char*>(RawVector((SEXP) lSampler[strSamples]).begin()),
				    Rf_isNull(lSampler[strExtent]) ? nullptr : as<NumericVector>(lSampler[strExtent]).begin(),
				    Rf_isNull(lSampler[strIndex]) ? nullptr : as<NumericVector>(lSampler[strIndex]).begin(),
				    bagging);
}


unique_ptr<SamplerBridge> SamplerR::unwrapFac(const List& lSampler,
					      bool bagging) {
    IntegerVector yTrain((SEXP) lSampler[strYTrain]);
    IntegerVector yZero = yTrain - 1;
    vector<unsigned int> yTrainCore(yZero.begin(), yZero.end());

    return make_unique<SamplerBridge>(move(yTrainCore),
				      as<CharacterVector>(yTrain.attr("levels")).length(),
				      as<IndexT>(lSampler[strNSamp]),
				      as<unsigned int>(lSampler[strNTree]),
				      as<string>(as<RawVector>(lSampler[strSamples]).attr("class")) == "nux",  
				      Rf_isNull(lSampler[strSamples]) ? nullptr : reinterpret_cast<unsigned char*>(RawVector((SEXP) lSampler[strSamples]).begin()),
				      Rf_isNull(lSampler[strExtent]) ? nullptr : as<NumericVector>(lSampler[strExtent]).begin(),
				      Rf_isNull(lSampler[strIndex]) ? nullptr : as<NumericVector>(lSampler[strIndex]).begin(),
				      bagging);
}
