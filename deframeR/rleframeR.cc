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


RcppExport SEXP PresortNum(SEXP sFrame) {
  BEGIN_RCPP

  List frame(sFrame);
  if (!frame.inherits("Frame")) {
    stop("Expecting Frame");
  }
  return RLEFrameR::presortNum(frame);

  END_RCPP
}


RcppExport SEXP PresortDF(SEXP sDF) {
  BEGIN_RCPP
    
  List df(sDF);
  return RLEFrameR::presortDF(df);
  
  END_RCPP
}


List RLEFrameR::presortDF(const DataFrame& df) {
  BEGIN_RCPP

  auto rleCresc = make_unique<RLECresc>(df.nrow(), df.length());

  // 'df' already screened for factor and numeric only.  In particular,
  // integer values are 32-bit nonnegative.
  // Caches column base addresses to avoid using Rcpp in OpenMP
  // loop.
  // N.B.:  According to Rcpp documentation, this style of Vector
  // constructor acts as a wrapper, rather than copying to memory.
  // Otherwise, we would be caching the addresses of temporaries.
  vector<void*> colBase(df.length());
  for (unsigned int predIdx = 0; predIdx < df.length(); predIdx++) {
    if (Rf_isFactor(df[predIdx])) {
      rleCresc->setFactor(predIdx, true);
      colBase[predIdx] = IntegerVector(df[predIdx]).begin();
    }
    else {
      rleCresc->setFactor(predIdx, false);
      colBase[predIdx] = NumericVector(df[predIdx]).begin();
    }
  }

  rleCresc->encodeFrame(colBase);

  return wrap(rleCresc.get());

  END_RCPP
}


List RLEFrameR::presortNum(const List& frame) {
  BEGIN_RCPP

  unsigned int nPredNum = as<unsigned int>(frame["nPredNum"]);
  auto rleCresc = make_unique<RLECresc>(as<size_t>(frame["nRow"]),
					nPredNum);

  vector<vector<double>> numVal;
  // Numeric block currently either dense or sparse, with a run-length
  // characterization.
  List blockNumIP((SEXP) frame["blockNumRLE"]);
  if (blockNumIP.length() > 0) {
    if (!blockNumIP.inherits("BlockNumIP")) {
      stop("Expecting BlockNumIP");
    }
    NumericVector valNumFE((SEXP) blockNumIP["valNum"]);
    vector<double> valNum(valNumFE.begin(), valNumFE.end());
    IntegerVector rowStartFE((SEXP) blockNumIP["rowStart"]);
    vector<size_t> rowStart(rowStartFE.begin(), rowStartFE.end());
    IntegerVector runLengthFE((SEXP) blockNumIP["runLength"]);
    vector<size_t> runLength(runLengthFE.begin(), runLengthFE.end());
    rleCresc->encodeFrameNum(move(valNum), move(rowStart), move(runLength));
  }
  else {
    rleCresc->encodeFrameNum(NumericMatrix((SEXP) frame["blockNum"]).begin());
  }

  return wrap(rleCresc.get());

  END_RCPP
}


List RLEFrameR::wrap(const RLECresc* rleCresc) {
  BEGIN_RCPP

  List setOut = List::create(
                             _["rankedFrame"] =  wrapRF(rleCresc),
                             _["numRanked"] = wrapNum(rleCresc),
			     _["facRanked"] = wrapFac(rleCresc)
                             );
  setOut.attr("class") = "RLEFrame";
  return setOut;

  END_RCPP
}


List RLEFrameR::wrapFac(const RLECresc* rleCresc) {
  BEGIN_RCPP

  vector<size_t> facHeight;
  vector<unsigned int> facValOut;
  for (auto facPred : rleCresc->getValFac()) {
    for (auto val : facPred) {
      facValOut.push_back(val);
    }
    facHeight.push_back(facValOut.size());
  }
  

  // Ranked numerical values for splitting-value interpolation.
  //
  List facRanked = List::create(
                                _["facVal"] = facValOut,
                                _["facHeight"] = facHeight
                                );
  facRanked.attr("class") = "FacRanked";

  return facRanked;
  END_RCPP
}


List RLEFrameR::wrapNum(const RLECresc* rleCresc) {
  BEGIN_RCPP

  vector<size_t> numHeight;
  vector<double> numValOut;
  for (auto numPred : rleCresc->getValNum()) {
    for (auto val : numPred) {
      numValOut.push_back(val);
    }
    numHeight.push_back(numValOut.size());
  }
  

  // Ranked numerical values for splitting-value interpolation.
  //
  List numRanked = List::create(
                                _["numVal"] = numValOut,
                                _["numHeight"] = numHeight
                                );
  numRanked.attr("class") = "NumRanked";

  return numRanked;
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
				  _["nRow"] = rleCresc->getNRow(),
				  _["runVal"] = valOut,
				  _["runLength"] = lengthOut,
				  _["runRow"] = rowOut,
				  _["rleHeight"] = rleHeight,
				  _["predForm"] = rleCresc->dumpPredForm()
                              );
  rankedFrame.attr("class") = "RankedFrame";
  return rankedFrame;
  END_RCPP
}


unique_ptr<RLEFrame> RLEFrameR::unwrap(const List& sRLEFrame) {
  List rleList(sRLEFrame);

  List blockNum = checkNumRanked((SEXP) rleList["numRanked"]);
  NumericVector numVal(Rf_isNull(blockNum["numVal"]) ? NumericVector(0) : NumericVector((SEXP) blockNum["numVal"]));
  IntegerVector numHeight(Rf_isNull(blockNum["numHeight"]) ? IntegerVector(0) : IntegerVector((SEXP) blockNum["numHeight"]));

  List blockFac = checkFacRanked((SEXP) rleList["facRanked"]);
  IntegerVector facVal(Rf_isNull(blockFac["facVal"]) ? NumericVector(0) : NumericVector((SEXP) blockFac["facVal"]));
  IntegerVector facHeight(Rf_isNull(blockFac["facHeight"]) ? IntegerVector(0) : IntegerVector((SEXP) blockFac["facHeight"]));

  List rankedFrame((SEXP) rleList["rankedFrame"]);
  if (rankedFrame.inherits("RankedFrame")) {
    return unwrapFrame(rankedFrame, numVal, numHeight, facVal, facHeight);
  }
  else {
    stop("Expecting RankedFrame");
  }
}


unique_ptr<RLEFrame> RLEFrameR::unwrapFrame(const List& rankedFrame,
					    const NumericVector& numValFE,
					    const IntegerVector& numHeightFE,
					    const IntegerVector& facValFE,
					    const IntegerVector& facHeightFE) {
  IntegerVector valFE((SEXP) rankedFrame["runVal"]);
  vector<size_t> runVal(valFE.begin(), valFE.end());
  IntegerVector lengthFE((SEXP) rankedFrame["runLength"]);
  vector<size_t> runLength(lengthFE.begin(), lengthFE.end());
  IntegerVector rowFE((SEXP) rankedFrame["runRow"]);
  vector<size_t> runRow(rowFE.begin(), rowFE.end());
  IntegerVector heightFE((SEXP) rankedFrame["rleHeight"]);
  vector<size_t> rleHeight(heightFE.begin(), heightFE.end());
  IntegerVector predFormFE((SEXP) rankedFrame["predForm"]);
  vector<PredictorForm> predForm;
  for (auto form : predFormFE) {
    predForm.push_back(static_cast<PredictorForm>(form));
  }
  
  vector<double> numVal(numValFE.begin(), numValFE.end());
  vector<size_t> numHeight(numHeightFE.begin(), numHeightFE.end());
  vector<unsigned int> facVal(facValFE.begin(), facValFE.end());
  vector<size_t> facHeight(facHeightFE.begin(), facHeightFE.end());

  size_t nRow(as<size_t>((SEXP) rankedFrame["nRow"]));
  return make_unique<RLEFrame>(nRow,
			       move(predForm),
			       move(runVal),
			       move(runLength),
			       move(runRow),
			       move(rleHeight),
			       move(numVal),
			       move(numHeight),
			       move(facVal),
			       move(facHeight));
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


List RLEFrameR::checkFacRanked(SEXP sFacRanked) {
  BEGIN_RCPP

  List facRanked(sFacRanked);
  if (!facRanked.inherits("FacRanked")) {
    stop("Expecting FacRanked");
  }

  return facRanked;

 END_RCPP
}
