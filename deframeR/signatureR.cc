// Copyright (C)  2012-2023   Mark Seligman
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

const string SignatureR::strColName = "colNames";
const string SignatureR::strRowName = "rowNames";
const string SignatureR::strPredLevel = "level";
const string SignatureR::strPredFactor = "factor";
const string SignatureR::strPredType = "predForm";
const string SignatureR::strFactorType = "factor";
const string SignatureR::strNumericType = "numeric";


RcppExport SEXP columnOrder(const SEXP sDF,
			    const SEXP sSigTrain,
			    const SEXP sKeyed) {
  BEGIN_RCPP

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
  END_RCPP
}


List SignatureR::wrapSparse(unsigned int nPred,
			    bool isFactor,
			    const CharacterVector& colNames,
			    const CharacterVector& rowNames) {
  BEGIN_RCPP

  return wrapMixed(nPred, rep(CharacterVector(isFactor ? strFactorType : strNumericType), nPred), List::create(0), List::create(0), colNames, rowNames);

  END_RCPP
}


List SignatureR::wrapNumeric(const NumericMatrix& blockNum) {
  BEGIN_RCPP

  unsigned int nPred = blockNum.ncol();
  return wrapMixed(nPred,
		   rep(CharacterVector(strNumericType), nPred),
		   List::create(0),
		   List::create(0),
		   Rf_isNull(colnames(blockNum)) ? CharacterVector(0) : colnames(blockNum),
		   Rf_isNull(rownames(blockNum)) ? CharacterVector(0) : rownames(blockNum));
  END_RCPP
}


List SignatureR::wrapFactor(const IntegerMatrix& blockFac) {
  BEGIN_RCPP

  unsigned int nPred = blockFac.ncol();
  return wrapMixed(nPred,
		   rep(CharacterVector(strFactorType), nPred),
		   List::create(0),
		   List::create(0),
		   Rf_isNull(colnames(blockFac)) ? CharacterVector(0) : colnames(blockFac),
		   Rf_isNull(rownames(blockFac)) ? CharacterVector(0) : rownames(blockFac));
  END_RCPP
}


List SignatureR::wrapMixed(unsigned int nPred,
			   const CharacterVector& predClass,
			   const List& level,
			   const List& factor,
			   const CharacterVector& colNames,
			   const CharacterVector& rowNames) {
  BEGIN_RCPP

  List signature =
    List::create(_[strPredType] = predClass,
                 _[strPredLevel] = level,
		 _[strPredFactor] = factor,
                 _[strColName] = colNames,
                 _[strRowName] = rowNames
                 );
  signature.attr("class") = "Signature";

  return signature;
  END_RCPP
}


List SignatureR::wrapDF(const DataFrame& df,
			const CharacterVector& predClass,
			const List& lLevel,
			const List& lFactor) {
  BEGIN_RCPP

  return wrapMixed(df.length(),
		   predClass,
		   lLevel,
		   lFactor,
		   Rf_isNull(as<CharacterVector>(df.names())) ? CharacterVector(0) : df.names(),
		   Rf_isNull(rownames(df)) ? CharacterVector(0) : rownames(df));

  END_RCPP
}

bool SignatureR::checkKeyable(const List& lSignature) {
  BEGIN_RCPP

  bool keyable = true;

  CharacterVector nullVec(as<CharacterVector>(lSignature[strColName]).length());
  if (Rf_isNull(lSignature[strColName])) {
    keyable = false;
    warning("No signature column names:  keyed access not supported");
  }
  else if (!is_true(all(as<CharacterVector>(lSignature[strColName]) != nullVec))) {
    keyable = false;
    warning("Empty signature column names:  keyed access not supported");
  }
  else if (as<CharacterVector>(lSignature[strColName]).length() != as<CharacterVector>(unique(as<CharacterVector>(lSignature[strColName]))).length()) {
    keyable = false;
    warning("Duplicate signature column names:  keyed access not supported");
  }
  
  return keyable;
  END_RCPP
}



/**
   @brief Unwraps field values useful for prediction.
 */
CharacterVector SignatureR::unwrapRowNames(const List& lDeframe) {
  BEGIN_RCPP
  checkFrame(lDeframe);
  List signature = checkSignature(lDeframe);

  if (Rf_isNull(signature[strRowName])) {
    return CharacterVector(0);
  }
  else {
    return CharacterVector((SEXP) signature[strRowName]);
  }
  END_RCPP
}


CharacterVector SignatureR::unwrapColNames(const List& lDeframe) {
  BEGIN_RCPP
  checkFrame(lDeframe);
  List signature = checkSignature(lDeframe);

  if (Rf_isNull(signature[strColName])) {
    return CharacterVector(0);
  }
  else {
    return CharacterVector((SEXP) signature[strColName]);
  }
  END_RCPP
}


SEXP SignatureR::checkSignature(const List &lDeframe) {
  BEGIN_RCPP
  List signature((SEXP) lDeframe["signature"]);
  if (!signature.inherits("Signature")) {
    stop("Expecting Signature");
  }

  return signature;
  END_RCPP
}


List SignatureR::unwrapLevel(const List& sTrain) {
 List sSignature(checkSignature(sTrain));
 return as<List>(sSignature[strPredLevel]);
}


List SignatureR::unwrapFactor(const List& sTrain) {
  List sSignature(checkSignature(sTrain));
  return as<List>(sSignature[strPredFactor]);
}


SignatureExpand SignatureExpand::unwrap(const List& lTrain) {
  List lSignature(SignatureR::checkSignature(lTrain));
  return SignatureExpand(as<List>(lSignature[SignatureR::strPredLevel]), as<List>(lSignature[SignatureR::strPredFactor]), as<CharacterVector>(lSignature[SignatureR::strColName]));
}


SignatureExpand::SignatureExpand(const List& level,
				 const List& factor,
				 const StringVector& names) {
  this->level = level;
  this->factor = factor;
  this->names = names;
}


SEXP SignatureR::checkFrame(const List &lDeframe) {
  BEGIN_RCPP
  if (!lDeframe.inherits("Deframe")) {
    stop("Expecting Derame");
  }

  END_RCPP
}


SEXP SignatureR::checkTypes(const List& lSigTrain,
			    const CharacterVector& predClass) {
  BEGIN_RCPP

  CharacterVector formTrain(as<CharacterVector>(lSigTrain[strPredType]));
  if (!is_true(all(formTrain == predClass))) {
    stop("Training, prediction data types do not match");
  }

  END_RCPP
}
