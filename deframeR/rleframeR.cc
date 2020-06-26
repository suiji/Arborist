// Copyright (C)  2012-2020   Mark Seligman
//
// This file is part of framemapR.
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
   @file rleframeR.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "rleframeR.h"


RcppExport SEXP Presort(SEXP sFrame) {
  BEGIN_RCPP

  List frame(sFrame);
  if (!frame.inherits("Frame")) {
    stop("Expecting Frame");
  }
  return RLEFrameR::presort(frame);

  END_RCPP
}


List RLEFrameR::presort(const List& frame) {
  BEGIN_RCPP

  auto rleCresc = make_unique<RLECresc>(as<unsigned int>(frame["nRow"]),
                                        as<unsigned int>(frame["nPredNum"]),
                                        as<unsigned int>(frame["nPredFac"]));
  // Numeric block currently either dense or sparse, with a run-length
  // characterization.
  List blockNumIP((SEXP) frame["blockNumRLE"]);
  if (blockNumIP.length() > 0) {
    if (!blockNumIP.inherits("BlockNumIP")) {
      stop("Expecting BlockNumIP");
    }
    rleCresc->numSparse(NumericVector((SEXP) blockNumIP["valNum"]).begin(),
     (unsigned int*) IntegerVector((SEXP) blockNumIP["rowStart"]).begin(),
     (unsigned int*) IntegerVector((SEXP) blockNumIP["runLength"]).begin()
                     );
  }
  else {
    rleCresc->numDense(NumericMatrix((SEXP) frame["blockNum"]).begin());
  }

  // Factor block currently dense.
  rleCresc->facDense((unsigned int*) IntegerMatrix((SEXP) frame["blockFac"]).begin());

  return wrap(rleCresc.get());

  END_RCPP
}


List RLEFrameR::wrap(const RLECresc *rleCresc, bool packed) {
  BEGIN_RCPP

  List setOut = List::create(
                             _["cardinality"] = rleCresc->getCardinality(),
                             _["rankedFrame"] = packed ? wrapRFPacked(rleCresc) : wrapRF(rleCresc),
                             _["numRanked"] = wrapNR(rleCresc)
                             );
  setOut.attr("class") = "RLEFrame";
  return setOut;

  END_RCPP
}


List RLEFrameR::wrapNR(const RLECresc* rleCresc) {
  BEGIN_RCPP
  // Ranked numerical values for splitting-value interpolation.
  //
  List numRanked = List::create(
                                _["numVal"] = rleCresc->getNumVal(),
                                _["numOff"] = rleCresc->getNumOff()
                                );
  numRanked.attr("class") = "NumRanked";

  return numRanked;
  END_RCPP
}


List RLEFrameR::wrapRFPacked(const RLECresc* rleCresc) {
  BEGIN_RCPP
  RawVector rleOut(rleCresc->getRLEBytes());
  vector<size_t> heightInternal(rleCresc->getHeight());
  NumericVector rleHeight(heightInternal.begin(), heightInternal.end());
  rleCresc->dumpRaw(rleOut.begin());
  List rankedFrame = List::create(
                              _["unitSize"] = RLECresc::unitSize(),
                              _["rle"] = rleOut,
			      _["rleHeight"] = rleHeight
                              );
  rankedFrame.attr("class") = "RankedFramePacked";
  return rankedFrame;
  END_RCPP
}


List RLEFrameR::wrapRF(const RLECresc* rleCresc) {
  BEGIN_RCPP

  vector<size_t> rleHeight(rleCresc->getHeight());
  size_t height = rleHeight.back();
  vector<size_t> valOut(height);
  vector<size_t> lengthOut(height);
  vector<size_t> rowOut(height);
  rleCresc->dump(valOut, lengthOut, rowOut);
  List rankedFrame = List::create(
                              _["runVal"] = valOut,
			      _["runLength"] = lengthOut,
			      _["runRow"] = rowOut,
			      _["rleHeight"] = rleHeight
                              );
  rankedFrame.attr("class") = "RankedFrame";
  return rankedFrame;
  END_RCPP
}


unique_ptr<RLEFrame> RLEFrameR::unwrap(const List& sRLEFrame, size_t nRow) {
  List rleList(sRLEFrame);

  IntegerVector cardFE(Rf_isNull(rleList["cardinality"]) ? IntegerVector(0) : IntegerVector((SEXP) rleList["cardinality"]));
  vector<unsigned int> cardinality(cardFE.begin(), cardFE.end());

  List blockNum = checkNumRanked((SEXP) rleList["numRanked"]);
  NumericVector numVal(Rf_isNull(blockNum["numVal"]) ? NumericVector(0) : NumericVector((SEXP) blockNum["numVal"]));
  NumericVector numOff(Rf_isNull(blockNum["numOff"]) ? NumericVector(0) : NumericVector((SEXP) blockNum["numOff"]));

  List rankedFrame((SEXP) rleList["rankedFrame"]);
  if (rankedFrame.inherits("RankedFramePacked")) {
    return unwrapFramePacked(rankedFrame, cardinality, nRow, numVal, numOff);
  }
  else {
    stop("Expecting RankedFrame");
  }
}


unique_ptr<RLEFrame> RLEFrameR::unwrapFramePacked(const List& rankedFrame,
						  vector<unsigned int>& cardinality,
						  size_t nRow,
						  const NumericVector& numVal_,
						  const NumericVector& numOff_) {
  RawVector rleRaw((SEXP) rankedFrame["rle"]);
  NumericVector rleFE((SEXP) rankedFrame["rleHeight"]);
  vector<size_t> rleHeight(rleFE.begin(), rleFE.end());

  vector<double> numVal(numVal_.begin(), numVal_.end());
  vector<size_t> numOff(numOff_.begin(), numOff_.end());
  return make_unique<RLEFrame>(nRow,
			       cardinality,
			       (const RLEVal<unsigned int>*) &rleRaw[0],
			       rleHeight,
			       numVal,
			       numOff);;
}


List RLEFrameR::checkRankedFrame(SEXP sRankedFrame) {
  BEGIN_RCPP

  List rankedFrame(sRankedFrame);
  if (!rankedFrame.inherits("RankedFrame")) {
    stop("Expecting RankedFrame");
  }
  if (Rf_isNull(rankedFrame["rle"])) {
    stop("Empty run encoding");
  }

  // Ensures compatibility across systems.
  if (as<int>(rankedFrame["unitSize"]) != RLECresc::unitSize()) {
    stop("Packing unit mismatch");
  }
                  
  return rankedFrame;

 END_RCPP
}


List RLEFrameR::checkNumRanked(SEXP sNumRanked) {
  BEGIN_RCPP

  List numRanked(sNumRanked);
  if (!numRanked.inherits("NumRanked")) {
    stop("Expecting NumRanked");
  }

  return numRanked;

 END_RCPP
}
