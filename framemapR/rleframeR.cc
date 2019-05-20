// Copyright (C)  2012-2019   Mark Seligman
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
                                       as<unsigned int>(frame["nPredFac"])
                                       );
  // Numeric block currently either dense or sparse, with a run-length
  // characterization.
  List blockNumIP((SEXP) frame["blockNumSparse"]);
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


List RLEFrameR::wrap(const RLECresc *rleCresc) {
  BEGIN_RCPP

  // Ranked numerical values for splitting-value interpolation.
  //
  List numRanked = List::create(
                                _["numVal"] = rleCresc->getNumVal(),
                                _["numOff"] = rleCresc->getNumOff()
                                );
  numRanked.attr("class") = "NumRanked";

  List rankedFrame = List::create(
                              _["row"] = rleCresc->getRow(),
                              _["rank"] = rleCresc->getRank(),
                              _["runLength"] = rleCresc->getRunLength()
                              );
  rankedFrame.attr("class") = "RankedFrame";

  List setOut = List::create(
                             _["cardinality"] = rleCresc->getCardinality(),
                             _["rankedFrame"] = move(rankedFrame),
                             _["numRanked"] = move(numRanked)
                             );
  setOut.attr("class") = "RLEFrame";
  return setOut;

  END_RCPP
}


unique_ptr<RLEFrame> RLEFrameR::factory(SEXP sRLEFrame,
                                        unsigned int nRow) {
  List rleList(sRLEFrame);
  List rankedFrame = checkRankedFrame(rleList["rankedFrame"]);
  List blockNum = checkNumRanked((SEXP) rleList["numRanked"]);
  auto rleFrame = factory(Rf_isNull(rleList["cardinality"]) ? IntegerVector(0) : IntegerVector((SEXP) rleList["cardinality"]), 
                          nRow,
                          IntegerVector((SEXP) rankedFrame["row"]),
                          IntegerVector((SEXP) rankedFrame["rank"]),
                          IntegerVector((SEXP) rankedFrame["runLength"]),
                          Rf_isNull(blockNum["numVal"]) ? NumericVector(0) : NumericVector((SEXP) blockNum["numVal"]),
                          Rf_isNull(blockNum["numOff"]) ? IntegerVector(0) : IntegerVector((SEXP) blockNum["numOff"]));
  return rleFrame;
}


unique_ptr<RLEFrame> RLEFrameR::factory(const IntegerVector& card,
                                        size_t nRow,
                                        const IntegerVector& row,
                                        const IntegerVector& rank,
                                        const IntegerVector& runLength,
                                        const NumericVector& numVal,
                                        const IntegerVector& numOff) {
  vector<unsigned int> cardinality(card.begin(), card.end());
  return make_unique<RLEFrame>(nRow,
                               cardinality,
                               row.length(),
                               (const unsigned int*) &row[0],
                               (const unsigned int*) &rank[0],
                               (const unsigned int*) &runLength[0],
                               (unsigned int) numOff.size(),
                               (const double*) &numVal[0],
                               (const unsigned int*) &numOff[0]);
}


List RLEFrameR::checkRankedFrame(SEXP sRankedFrame) {
  BEGIN_RCPP

  List rankedFrame(sRankedFrame);
  if (!rankedFrame.inherits("RankedFrame")) {
    stop("Expecting RankedFrame");
  }
  if (Rf_isNull(rankedFrame["row"])) {
    stop("Empty run encoding");
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
