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
   @file signatureR.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "signatureR.h"

const string SignatureR::strClassName = "Signature";
const string SignatureR::strColName = "colNames";
const string SignatureR::strRowName = "rowNames";
const string SignatureR::strPredLevel = "level";
const string SignatureR::strPredFactor = "factor";
const string SignatureR::strPredType = "predForm";
const string SignatureR::strFactorType = "factor";
const string SignatureR::strNumericType = "numeric";

// [[Rcpp::export]]
RcppExport SEXP columnOrder(const SEXP sDF,
			    const SEXP sSigTrain,
			    const SEXP sKeyed) {
  DataFrame df(as<DataFrame>(sDF));
  if (!Rf_isNull(sSigTrain) && as<bool>(sKeyed)) {
    List lSigTrain(sSigTrain);
    if (SignatureR::checkKeyable(List(sSigTrain))) {
      // Matches signature columns within new frame and caches match indices.
      // Bails if any are not present, but does not search for duplicates.
      IntegerVector colMatch(match(as<CharacterVector>(lSigTrain[SignatureR::strColName]), as<CharacterVector>(df.names())));
      if (is_true(any(is_na(colMatch)))) {
	warning("Some signature names do not appear in the new frame:  keyed access not supported");
      }
      else {
	return wrap(colMatch);
      }
    }
  }
  return wrap(seq(1, df.length()));
}


// [[Rcpp::export]]
bool SignatureR::checkKeyable(const List& lSignature) {
  if (Rf_isNull(lSignature[strColName])) {
    warning("No signature column names:  keyed access not supported");
    return false;
  }

  bool keyable = false;
  CharacterVector colNames(as<CharacterVector>(lSignature[strColName]));
  CharacterVector nullVec(colNames.length());
  if (!is_true(all(colNames != nullVec))) {
    keyable = false;
    warning("Empty signature column names:  keyed access not supported");
  }
  else if (colNames.length() != as<CharacterVector>(unique(colNames)).length()) {
    keyable = false;
    warning("Duplicate signature column names:  keyed access not supported");
  }
  
  return keyable;
}


IntegerVector SignatureR::predMap(const List& lTrain) {
  List lSignature(getSignature(lTrain));
  CharacterVector predType(lSignature[strPredType]);
  IntegerVector packed2Idx(predType.length());
  unsigned int idxNum = 0;
  unsigned int idxFac = predType.length() - nFactor(lTrain);
  for (unsigned int i = 0; i != predType.length(); i++) {
    if (predType[i] == strNumericType) {
      packed2Idx[idxNum++] = i;
    }
    else if (predType[i] == strFactorType) {
      packed2Idx[idxFac++] = i;
    }
    else
      stop("Unexpected predictor type.");
  }
  
  return packed2Idx;
}


unsigned int SignatureR::nPred(const List& lTrain) {
  List lSignature(getSignature(lTrain));
  CharacterVector predType(lSignature[strPredType]);
  return predType.length();
}


// [[Rcpp::export]]
List SignatureR::wrapSparse(unsigned int nPred,
			    bool isFactor,
			    const CharacterVector& colNames,
			    const CharacterVector& rowNames) {
  return wrapMixed(nPred, rep(CharacterVector(isFactor ? strFactorType : strNumericType), nPred), List::create(0), List::create(0), colNames, rowNames);
}


// [Rcpp::export]]
List SignatureR::wrapNumeric(const NumericMatrix& blockNum) {
  unsigned int nPred = blockNum.ncol();
  return wrapMixed(nPred,
		   rep(CharacterVector(strNumericType), nPred),
		   List::create(0),
		   List::create(0),
		   Rf_isNull(colnames(blockNum)) ? CharacterVector(0) : colnames(blockNum),
		   Rf_isNull(rownames(blockNum)) ? CharacterVector(0) : rownames(blockNum));
}


// [[Rcpp::export]]
List SignatureR::wrapFactor(const IntegerMatrix& blockFac) {
  unsigned int nPred = blockFac.ncol();
  return wrapMixed(nPred,
		   rep(CharacterVector(strFactorType), nPred),
		   List::create(0),
		   List::create(0),
		   Rf_isNull(colnames(blockFac)) ? CharacterVector(0) : colnames(blockFac),
		   Rf_isNull(rownames(blockFac)) ? CharacterVector(0) : rownames(blockFac));
}


// [[Rcpp::export]]
List SignatureR::wrapMixed(unsigned int nPred,
			   const CharacterVector& predClass,
			   const List& level,
			   const List& factor,
			   const CharacterVector& colNames,
			   const CharacterVector& rowNames) {
  List signature =
    List::create(_[strPredType] = predClass,
                 _[strPredLevel] = level,
		 _[strPredFactor] = factor,
                 _[strColName] = colNames,
                 _[strRowName] = rowNames
                 );
  signature.attr("class") = strClassName;

  return signature;
}


// [[Rcpp::export]]
List SignatureR::wrapDF(const DataFrame& df,
			const CharacterVector& predClass,
			const List& lLevel,
			const List& lFactor) {
  return wrapMixed(df.length(),
		   predClass,
		   lLevel,
		   lFactor,
		   Rf_isNull(as<CharacterVector>(df.names())) ? CharacterVector(0) : df.names(),
		   Rf_isNull(rownames(df)) ? CharacterVector(0) : rownames(df));
}


/**
   @brief Unwraps field values useful for prediction.
 */
// [[Rcpp::export]]
CharacterVector SignatureR::unwrapRowNames(const List& lDeframe) {
  if (!checkFrame(lDeframe))
    stop("Expecting Deframe object");

  return unwrapName(getSignature(lDeframe), strRowName);
}


// [[Rcpp::export]]
CharacterVector SignatureR::unwrapColNames(const List& lDeframe) {
  if (!checkFrame(lDeframe))
    stop("Expecting Deframe object.");

  return unwrapName(getSignature(lDeframe), strColName);
}


CharacterVector SignatureR::unwrapName(const List& signature,
				       const string& name) {
  return Rf_isNull(signature[name]) ? CharacterVector(0) : CharacterVector((SEXP) signature[name]);
}


List SignatureR::getSignature(const List& lParent) {
  List signature((SEXP) lParent["signature"]);
  if (!signature.inherits("Signature")) {
    stop("Expecting Signature");
  }

  return signature;
}


List SignatureR::unwrapLevel(const List& lDeframe) {
  List lSignature = getSignature(lDeframe);
 return as<List>(lSignature[strPredLevel]);
}


List SignatureR::getFactor(const List& lDeframe) {
  List lSignature = getSignature(lDeframe);
  return as<List>(lSignature[strPredFactor]);
}


List SignatureR::getLevel(const List& lParent) {
  List lSignature = getSignature(lParent);
  return as<List>(lSignature[strPredLevel]);
}


unsigned int SignatureR::nFactor(const List& lParent) {
  return getLevel(lParent).length();
}


bool SignatureR::checkFrame(const List &lDeframe) {
  return lDeframe.inherits("Deframe");
}


bool SignatureR::checkTypes(SEXP sSigTrain,
			    const CharacterVector& predClass) {
  if (!Rf_isNull(sSigTrain)) {
    List lSigTrain(sSigTrain);
    CharacterVector formTrain(as<CharacterVector>(lSigTrain[strPredType]));
    if (!is_true(all(formTrain == predClass))) {
      return false;
    }
  }
  return true;
}
