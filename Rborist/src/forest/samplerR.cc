// Copyright (C)  2012-2021   Mark Seligman
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

#include "samplerR.h"
#include "samplerbridge.h"
#include "trainbridge.h"


bool SamplerR::trainThin = false;


SamplerR::SamplerR(unsigned int nTree_) :
  nTree(nTree_),
  samplerBlockHeight(vector<size_t>(nTree)),
  samplerBlockRaw(RawVector(0)) {
}


SamplerR::~SamplerR() {
}


SamplerCtgR::SamplerCtgR(const IntegerVector& yTrain_, unsigned int nTree_) :
  SamplerR(nTree_),
  yTrain(yTrain_) {
}


SamplerCtgR::~SamplerCtgR() {
}


SamplerRegR::SamplerRegR(const NumericVector& yTrain_, unsigned int nTree_) :
  SamplerR(nTree_),
  yTrain(yTrain_) {
}


SamplerRegR::~SamplerRegR() {
}


void SamplerR::consume(const TrainChunk* train,
		       unsigned int tIdx,
		       double scale) {
  train->writeSamplerBlockHeight(samplerBlockHeight, tIdx);

  // Writes BagSample records as raw.
  size_t blOff, bagBytes;
  if (!train->samplerBlockFits(samplerBlockHeight, tIdx, static_cast<size_t>(samplerBlockRaw.length()), blOff, bagBytes)) {
    samplerBlockRaw = move(resizeRaw(&samplerBlockRaw[0], blOff, bagBytes, scale));
  }
  train->dumpSamplerBlockRaw(&samplerBlockRaw[blOff]);
}


void SamplerCtgR::consume(const TrainChunk* train,
			  unsigned int tIdx,
			  double scale) {
  SamplerR::consume(train, tIdx, scale);
}


RawVector SamplerR::resizeRaw(const unsigned char* raw, size_t offset, size_t bytes, double scale) { // Assumes scale >= 1.0.
  RawVector temp(scale * (offset + bytes));
  for (size_t i = 0; i < offset; i++)
    temp[i] = raw[i];

  return temp;
}


List SamplerRegR::wrap() {
  BEGIN_RCPP

  List samplerReg = List::create(_["yTrain"] = yTrain,
				 _["samplerBlock"] = move(samplerBlockRaw),
				 _["nTree"] = nTree
			);
  samplerReg.attr("class") = "SamplerReg";
  
  return samplerReg;
  END_RCPP
}


List SamplerCtgR::wrap() {
  BEGIN_RCPP

  // Writes zero-based indices.
  //
  IntegerVector yZero = yTrain - 1;
  List samplerCtg = List::create(_["yTrain"] = yZero,
				 _["levels"] = yTrain.attr("levels"),
				 _["samplerBlock"] = move(samplerBlockRaw),
				 _["nTree"] = nTree
				 );
  samplerCtg.attr("class") = "SamplerCtg";

  return samplerCtg;

  END_RCPP
}


unsigned int SamplerCtgR::getNCtg(const List& lSampler) {
  return as<CharacterVector>(lSampler["levels"]).length();
}


unique_ptr<SamplerBridge> SamplerRegR::unwrap(const List& lTrain,
						 const List& lDeframe,
						 bool bagging) {
  List lSampler((SEXP) lTrain["sampler"]);
  if (bagging)
    checkOOB(lSampler, as<size_t>((SEXP) lDeframe["nRow"]));

  return unwrap(lSampler);
}


unique_ptr<SamplerBridge> SamplerCtgR::unwrap(const List& lTrain,
						 const List& lDeframe,
						 bool bagging) {
  List lSampler((SEXP) lTrain["sampler"]);
  if (bagging)
    checkOOB(lSampler, as<size_t>((SEXP) lDeframe["nRow"]));
  return unwrap(lSampler);
}


SEXP SamplerR::checkOOB(const List& lSampler, size_t nRow) {
  BEGIN_RCPP
  if (Rf_isNull(lSampler["samplerBlock"]))
    stop("Out-of-bag prediction requested with empty sampler.");

  R_xlen_t nObs = as<NumericVector>(lSampler["yTrain"]).length();

  if (static_cast<size_t>(nObs) != nRow)
    stop("Bag and prediction row counts do not agree.");

  END_RCPP
}


unique_ptr<SamplerBridge> SamplerRegR::unwrap(const List& lSampler) {
  if (!lSampler.inherits("SamplerReg")) {
    stop("Expecting SamplerReg");
  }

  NumericVector yTrain((SEXP) lSampler["yTrain"]);
  vector<double> yTrainCore(yTrain.begin(), yTrain.end());
  return make_unique<SamplerBridge>(move(yTrainCore),
				    as<unsigned int>(lSampler["nTree"]),
				    Rf_isNull(lSampler["samplerBlock"]) ? nullptr : (unsigned char*) RawVector((SEXP) lSampler["samplerBlock"]).begin());
}


unique_ptr<SamplerBridge> SamplerCtgR::unwrap(const List& lSampler) {
  if (!lSampler.inherits("SamplerCtg")) {
    stop("Expecting SamplerCtg");
  }

  IntegerVector yTrain((SEXP) lSampler["yTrain"]);
  vector<unsigned int> yTrainCore(yTrain.begin(), yTrain.end());
  return make_unique<SamplerBridge>(move(yTrainCore),
				    getNCtg(lSampler),
				    as<unsigned int>(lSampler["nTree"]),
				    Rf_isNull(lSampler["samplerBlock"]) ? nullptr : (unsigned char*) RawVector((SEXP) lSampler["samplerBlock"]).begin());
}

