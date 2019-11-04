// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of framemapR
//
// framemapR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// frameampR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with framemapR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file frame.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "signature.h"
#include "frame.h"
#include "block.h"

#include<memory>

RcppExport SEXP FrameReconcile(SEXP sXFac,
                               SEXP sPredMap,
                               SEXP sLevel,
                               SEXP sSigTrain) {
  BEGIN_RCPP

  IntegerVector predMap(sPredMap); // 0-based predictor offsets.
  List sigTrain(sSigTrain);
  IntegerVector predTrain(as<IntegerVector>(sigTrain["predMap"]));
  if (!is_true(all(predMap == predTrain))) {
    stop("Training, prediction data types do not match");
  }
  IntegerMatrix xFac(sXFac);// 0-based factor codes.
  List levelTest(sLevel);
  List levelTrain((SEXP) sigTrain["level"]); // SignatureRf::unwrapLevel()
  for (int col = 0; col < xFac.ncol(); col++) {
    CharacterVector colTest(as<CharacterVector>(levelTest[col]));
    CharacterVector colTrain(as<CharacterVector>(levelTrain[col]));
    if (is_true(any(colTest != colTrain))) {
      IntegerVector colMatch(match(colTest, colTrain));
      // Rcpp match() does not offer specification of 'na' subsititute.
      if (is_true(any(is_na(colMatch)))) {
        warning("Test data contains labels unseen by training:  employing proxy");
        colMatch = ifelse(is_na(colMatch), static_cast<int>(colTrain.length()) + 1, colMatch);
      }
      colMatch = colMatch - 1;  // Rcpp match() is one-based.

      IntegerVector colT(IntegerMatrix::Column(xFac(_, col)));
      xFac(_, col) = as<IntegerVector>(colMatch[colT]);
    }
  }

  return xFac;
  
  END_RCPP
}


RcppExport SEXP WrapFrame(SEXP sX,
                          SEXP sXNum,
                          SEXP sXFac,
                          SEXP sPredMap,
                          SEXP sFacCard,
                          SEXP sLevel) {
  BEGIN_RCPP

  NumericMatrix xNum(sXNum);
  IntegerVector facCard(sFacCard);
  IntegerMatrix xFac(sXFac);// 0-based factor codes.
  IntegerVector predMap(sPredMap); // 0-based predictor offsets.
  DataFrame x(sX);
  List frame = List::create(
                                _["blockNum"] = move(xNum),
                                _["nPredNum"] = xNum.ncol(),
                                _["blockNumRLE"] = List(), // For now.
                                _["blockFacRLE"] = R_NilValue, // For now.
                                _["blockFac"] = move(xFac),
                                _["nPredFac"] = xFac.ncol(),
                                _["nRow"] = x.nrow(),
                                _["facCard"] = facCard,
            _["signature"] = move(Signature::wrapSignature(predMap,
                                                            as<List>(sLevel),
                                                            Rf_isNull(colnames(x)) ? CharacterVector(0) : colnames(x),
                                                            Rf_isNull(rownames(x)) ? CharacterVector(0) : rownames(x)))
                                );
  frame.attr("class") = "Frame";

  return frame;
  END_RCPP
}


RcppExport SEXP FrameNum(SEXP sX) {
  NumericMatrix blockNum(sX);
  List frame = List::create(
        _["blockNum"] = blockNum,
        _["blockNumRLE"] = List(), // For now.
        _["blockFacRLE"] = R_NilValue, // For now.
        _["nPredNum"] = blockNum.ncol(),
        _["blockFac"] = IntegerMatrix(0),
        _["nPredFac"] = 0,
        _["nRow"] = blockNum.nrow(),
        _["facCard"] = IntegerVector(0),
        _["signature"] = move(Signature::wrapSignature(
                                        seq_len(blockNum.ncol()) - 1,
                                        List::create(0),
                                        Rf_isNull(colnames(blockNum)) ? CharacterVector(0) : colnames(blockNum),
                                        Rf_isNull(rownames(blockNum)) ? CharacterVector(0) : rownames(blockNum)))
                                );
  frame.attr("class") = "Frame";

  return frame;
}

// TODO:  Move column and row names to signature.
/**
   @brief Reads an S4 object containing (sparse) dgCMatrix.
 */
RcppExport SEXP FrameSparse(SEXP sX) {
  BEGIN_RCPP
  S4 spNum(sX);

  IntegerVector i;
  if (R_has_slot(sX, PROTECT(Rf_mkString("i")))) {
    i = spNum.slot("i");
  }
  IntegerVector j;
  if (R_has_slot(sX, PROTECT(Rf_mkString("j")))) {
    j = spNum.slot("j");
  }
  IntegerVector p;
  if (R_has_slot(sX, PROTECT(Rf_mkString("p")))) {
    p = spNum.slot("p");
  }

  if (!R_has_slot(sX, PROTECT(Rf_mkString("Dim")))) {
    stop("Expecting dimension slot");
  }
  if (!R_has_slot(sX, PROTECT(Rf_mkString("x")))) {
    stop("Pattern matrix:  NYI");
  }
  UNPROTECT(5);

  IntegerVector dim = spNum.slot("Dim"); // #row, #pred
  unsigned int nRow = dim[0];
  unsigned int nPred = dim[1];
  unique_ptr<BlockIPCresc<double> > rleCresc = make_unique<BlockIPCresc<double> >(nRow, nPred);

  // Divines the encoding format and packs appropriately.
  //
  if (i.length() == 0) {
    stop("Sparse form j/p:  NYI");
  }
  else if (p.length() == 0) {
    stop("Sparse form i/j:  NYI");
  }
  else if (j.length() == 0) {
    rleCresc->nzRow(&as<NumericVector>(spNum.slot("x"))[0], &i[0], &p[0]);
  }
  else {
    stop("Indeterminate sparse matrix format");
  }

  List blockNumIP = List::create(
                                 _["valNum"] = rleCresc->getVal(),
                                 _["rowStart"] = rleCresc->getRowStart(),
                                 _["runLength"] = rleCresc->getRunLength(),
                                 _["predStart"] = rleCresc->getPredStart());
  blockNumIP.attr("class") = "BlockNumIP";

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

  IntegerVector facCard(0);
  List frame = List::create(
        _["blockNum"] = NumericMatrix(0),
        _["nPredNum"] = nPred,
        _["blockNumRLE"] = move(blockNumIP),
        _["blockFacRLE"] = R_NilValue, // For now.
        _["blockFac"] = IntegerMatrix(0),
        _["nPredFac"] = 0,
        _["nRow"] = nRow,
        _["facCard"] = facCard,
        _["signature"] = move(Signature::wrapSignature(seq_len(nPred) - 1,
                                        List::create(0),
                                        colName,
                                        rowName))
                                );

  frame.attr("class") = "Frame";

  return frame;
  END_RCPP
}
