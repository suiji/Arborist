// Copyright (C)  2012-2024   Mark Seligman
//
// This file is part of deframeR.
//
// deframeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// deframeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with deframeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file rleframeR.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "rleframeR.h"
#include "signatureR.h"


List RLEFrameR::presortDF(const DataFrame& df, SEXP sSigTrain, SEXP sLevel, const CharacterVector& predClass) {
  BEGIN_RCPP

  IntegerMatrix factorRemap;
  if (!Rf_isNull(sSigTrain)) {
    SignatureR::checkTypes(List(sSigTrain), predClass);
    factorRemap = factorReconcile(df, List(sSigTrain), List(sLevel));
  }

  auto rleCresc = make_unique<RLECresc>(df.nrow(), df.length());

  // 'df' already screened for factor and numeric only.  In particular,
  // integer values are 32-bit nonnegative.
  // Caches column base addresses to avoid using Rcpp in OpenMP
  // loop.
  // N.B.:  According to Rcpp documentation, this style of Vector
  // constructor merely wraps a pointer and does not generate a copy.
  List lLevel(sLevel);
  unsigned int nFac = 0;
  vector<void*> colBase(df.length());
  for (R_xlen_t predIdx = 0; predIdx < df.length(); predIdx++) {
    if (Rf_isFactor(df[predIdx])) {
      rleCresc->setFactor(predIdx, as<CharacterVector>(lLevel[nFac]).length());
      colBase[predIdx] = !Rf_isNull(sSigTrain) ? IntegerVector(factorRemap(_, nFac)).begin() : IntegerVector(df[predIdx]).begin();
      nFac++;
    }
    else {
      rleCresc->setFactor(predIdx, 0);
      colBase[predIdx] = NumericVector(df[predIdx]).begin();
    }
  }

  rleCresc->encodeFrame(colBase);

  return wrap(rleCresc.get());

  END_RCPP
}


bool RLEFrameR::checkKeyable(const DataFrame& df,
			     const List& sigTrain) {
  BEGIN_RCPP

  return false;

  END_RCPP
}


IntegerMatrix RLEFrameR::factorReconcile(const DataFrame& df,
					 const List& lSigTrain,
					 const List& levelTest) {
  BEGIN_RCPP

  List levelTrain(as<List>(lSigTrain["level"]));
  IntegerMatrix mappedFactor(df.nrow(), levelTrain.length());
  unsigned int nFac = 0;
  for (int col = 0; col < df.length(); col++) {
    if (Rf_isFactor(df[col])) {
      mappedFactor(_, nFac) = columnReconcile(IntegerVector(df[col]), as<CharacterVector>(levelTest[nFac]), as<CharacterVector>(levelTrain[nFac]));
      nFac++;
    }
  }
  return mappedFactor;

  END_RCPP
}


IntegerVector RLEFrameR::columnReconcile(const IntegerVector& dfCol,
					 const CharacterVector& levelsTest,
					 const CharacterVector& levelsTrain) {
  BEGIN_RCPP
    
  if (is_true(any(levelsTest != levelsTrain))) {
    IntegerVector colMatch(match(levelsTest, levelsTrain));
    // Rcpp match() implementation does not suppport 'na' subsititute.
    if (is_true(any(is_na(colMatch)))) {
      warning("Test data contains labels absent from training:  employing proxy factor");
      colMatch = ifelse(is_na(colMatch), static_cast<int>(levelsTrain.length()) + 1, colMatch);
    }

    // N.B.:  Rcpp::match() indices are one-based.
    IntegerVector dfZero(dfCol - 1); // R factor indices are one-based.

    // Rcpp subscripting is zero-based.
    return as<IntegerVector>(colMatch[dfZero]);
  }
  else {
    return dfCol;
  }

  END_RCPP
}


List RLEFrameR::presortIP(const BlockIPCresc<double>* rleCrescIP, size_t nRow, unsigned int nPred) {
  BEGIN_RCPP

    auto rleCresc = make_unique<RLECresc>(nRow, nPred);

  vector<double> valNum(rleCrescIP->getVal());
  vector<size_t> rowStart(rleCrescIP->getRunStart());
  vector<size_t> runLength(rleCrescIP->getRunLength());
  rleCresc->encodeFrameNum(std::move(valNum), std::move(rowStart), std::move(runLength));

  return wrap(rleCresc.get());

  END_RCPP
}


List RLEFrameR::presortNum(const SEXP sX) {
  BEGIN_RCPP

  NumericMatrix x(sX);
  auto rleCresc = make_unique<RLECresc>(x.nrow(), x.ncol());
  rleCresc->encodeFrameNum(x.begin());
  return wrap(rleCresc.get());

  END_RCPP
}


List RLEFrameR::presortFac(const SEXP sX) {
  BEGIN_RCPP

  IntegerMatrix x(sX);
  auto rleCresc = make_unique<RLECresc>(x.nrow(), x.ncol());
  rleCresc->encodeFrameFac(reinterpret_cast<uint32_t*>(x.begin()));

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
				  _["topIdx"] = rleCresc->dumpTopIdx()
                              );
  rankedFrame.attr("class") = "RankedFrame";
  return rankedFrame;
  END_RCPP
}


unique_ptr<RLEFrame> RLEFrameR::unwrap(const List& lDeframe) {
  List rleList((SEXP) lDeframe["rleFrame"]);
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
  IntegerVector topIdxFE((SEXP) rankedFrame["topIdx"]);
  vector<unsigned int> topIdx;
  for (auto card : topIdxFE) {
    topIdx.push_back(card);
  }
  
  vector<double> numVal(numValFE.begin(), numValFE.end());
  vector<size_t> numHeight(numHeightFE.begin(), numHeightFE.end());
  vector<unsigned int> facVal(facValFE.begin(), facValFE.end());
  vector<size_t> facHeight(facHeightFE.begin(), facHeightFE.end());

  size_t nRow(as<size_t>((SEXP) rankedFrame["nRow"]));
  return make_unique<RLEFrame>(nRow,
			       std::move(topIdx),
			       std::move(runVal),
			       std::move(runLength),
			       std::move(runRow),
			       std::move(rleHeight),
			       std::move(numVal),
			       std::move(numHeight),
			       std::move(facVal),
			       std::move(facHeight));
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
