// Copyright (C)  2012-2022   Mark Seligman
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
   @file deframe.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "signature.h"
#include "deframe.h"
#include "block.h"
#include "rleframeR.h"

#include<memory>

RcppExport SEXP deframeDF(SEXP sDf,
			  SEXP sPredForm,
                          SEXP sLevel,
			  SEXP sFactor,
			  SEXP sSigTrain) {
  BEGIN_RCPP

  if (!Rf_isNull(sSigTrain)) {
    checkFrame(List(sSigTrain), CharacterVector(sPredForm));
  }

  DataFrame df(sDf);
  List deframe = List::create(
			      _["rleFrame"] = RLEFrameR::presortDF(df, sSigTrain, sLevel),
			      _["nRow"] = df.nrow(),
			      _["signature"] = Signature::wrap(df.length(),
							       CharacterVector(sPredForm),
							       List(sLevel),
							       List(sFactor),
							       Rf_isNull(df.names()) ? CharacterVector(0) : df.names(),
							       Rf_isNull(rownames(df)) ? CharacterVector(0) : rownames(df))
			      );
  deframe.attr("class") = "Deframe";
  return deframe;

  END_RCPP
}


SEXP checkFrame(const List& lSigTrain,
		const CharacterVector& predForm) {
  BEGIN_RCPP
    
  CharacterVector formTrain(as<CharacterVector>(lSigTrain["predForm"]));
  if (!is_true(all(predForm == formTrain))) {
    stop("Training, prediction data types do not match");
  }

  END_RCPP
}


RcppExport SEXP deframeFac(SEXP sX) {
  IntegerMatrix blockFac(sX);
  List deframe = List::create(
			      _["rleFrame"] = RLEFrameR::presortFac(blockFac),
			      _["nRow"] = blockFac.nrow(),
			      _["signature"] = Signature::wrapFac(blockFac.ncol(),
								  Rf_isNull(colnames(blockFac)) ? CharacterVector(0) : colnames(blockFac),

								  Rf_isNull(rownames(blockFac)) ? CharacterVector(0) : rownames(blockFac))
			      );

  deframe.attr("class") = "Deframe";
  return deframe;
}


RcppExport SEXP deframeNum(SEXP sX) {
  NumericMatrix blockNum(sX);
  List deframe = List::create(
			      _["rleFrame"] = RLEFrameR::presortNum(blockNum),
			      _["nRow"] = blockNum.nrow(),
			      _["signature"] = Signature::wrapNum(blockNum.ncol(),
								  Rf_isNull(colnames(blockNum)) ? CharacterVector(0) : colnames(blockNum),

								  Rf_isNull(rownames(blockNum)) ? CharacterVector(0) : rownames(blockNum))
			      );

  deframe.attr("class") = "Deframe";
  return deframe;
}


/**
   @brief Reads an S4 object containing (sparse) dgCMatrix.
 */
RcppExport SEXP deframeIP(SEXP sX) {
  BEGIN_RCPP
  S4 spNum(sX);

  // Divines the encoding format and packs appropriately.
  //
  IntegerVector i;
  if (R_has_slot(sX, PROTECT(Rf_mkString("i")))) {
    i = spNum.slot("i");
    if (i.length() == 0) {
      stop("Sparse form j/p:  NYI");
    }
  }

  
  IntegerVector j;
  if (R_has_slot(sX, PROTECT(Rf_mkString("j")))) {
    j = spNum.slot("j");
    if (j.length() != 0) {
      stop("Indeterminate sparse matrix format");
    }
  }

  IntegerVector p;
  if (R_has_slot(sX, PROTECT(Rf_mkString("p")))) {
    p = spNum.slot("p");
    if (p.length() == 0) {
      stop("Sparse form i/j:  NYI");
    }
  }

  
  if (!R_has_slot(sX, PROTECT(Rf_mkString("Dim")))) {
    stop("Expecting dimension slot");
  }
  if (!R_has_slot(sX, PROTECT(Rf_mkString("x")))) {
    stop("Pattern matrix:  NYI");
  }
  UNPROTECT(5);

  IntegerVector dim = spNum.slot("Dim"); // #row, #pred
  size_t nRow = dim[0];
  unsigned int nPred = dim[1];
  unique_ptr<BlockIPCresc<double> > blockIPCresc = make_unique<BlockIPCresc<double> >(nRow, nPred);

  vector<size_t> rowNZ(i.begin(), i.end());
  vector<size_t> idxPred(p.begin(), p.end());
  blockIPCresc->nzRow(&as<NumericVector>(spNum.slot("x"))[0], rowNZ, idxPred);

  List dimNames;
  CharacterVector rowName = CharacterVector(0);
  CharacterVector colName = CharacterVector(0);
  if (R_has_slot(sX, PROTECT(Rf_mkString("Dimnames")))) {
    dimNames = spNum.slot("Dimnames");
    if (!Rf_isNull(dimNames[0])) {
      rowName = dimNames[0];
    }
    if (!Rf_isNull(dimNames[1])) {
      colName = dimNames[1];
    }
  }
  UNPROTECT(1);

  List deframe = List::create(
			      _["rleFrame"] = RLEFrameR::presortIP(blockIPCresc.get(), nRow, nPred),
			      _["nRow"] = nRow,
			      _["signature"] = Signature::wrapNum(nPred, colName, rowName));
  deframe.attr("class") = "Deframe";
  return deframe;

  END_RCPP
}
