// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of ArboristBridgeR.
//
// ArboristBridgeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// ArboristBridgeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file frameblockBridge.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/


#include "framemapBridge.h"
#include "blockBridge.h"


RcppExport SEXP FrameReconcile(SEXP sXFac,
                               SEXP sPredMap,
                               SEXP sLv,
                               SEXP sSigTrain) {
  BEGIN_RCPP

  IntegerVector predMap(sPredMap); // 0-based predictor offsets.
  List sigTrain(sSigTrain);
  IntegerVector predTrain(as<IntegerVector>(sigTrain["predMap"]));
  if (!is_true(all(predMap == predTrain))) {
    stop("Training, prediction data types do not match");
  }
  IntegerMatrix xFac(sXFac);// 0-based factor codes.
  List levelTest(sLv);
  List levelTrain((SEXP) sigTrain["level"]);
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
                          SEXP sLv) {
  BEGIN_RCPP

  NumericMatrix xNum(sXNum);
  IntegerVector facCard(sFacCard);
  IntegerMatrix xFac(sXFac);// 0-based factor codes.
  IntegerVector predMap(sPredMap); // 0-based predictor offsets.
  DataFrame x(sX);
  List predBlock = List::create(
                                _["blockNum"] = move(xNum),
                                _["nPredNum"] = xNum.ncol(),
                                _["blockNumSparse"] = List(), // For now.
                                _["blockFacSparse"] = R_NilValue, // For now.
                                _["blockFac"] = move(xFac),
                                _["nPredFac"] = xFac.ncol(),
                                _["nRow"] = x.nrow(),
                                _["facCard"] = facCard,
            _["signature"] = move(FramemapBridge::wrapSignature(predMap,
                                                                as<List>(sLv),
                                                                colnames(x),
                                                                rownames(x)))
                                );
  predBlock.attr("class") = "PredBlock";

  return predBlock;
  END_RCPP
}


// Signature contains front-end decorations not exposed to the
  // core.
// Column and row names stubbed to zero-length vectors if null.
SEXP FramemapBridge::wrapSignature(const IntegerVector &predMap,
               const List &level,
               const CharacterVector &colNames,
               const CharacterVector &rowNames) {
  BEGIN_RCPP
  List signature =
    List::create(
                 _["predMap"] = predMap,
                 _["level"] = level,
                 _["colNames"] = Rf_isNull(colNames) ? CharacterVector(0) : colNames,
                 _["rowNames"] = Rf_isNull(rowNames) ? CharacterVector(0) : rowNames
                 );
  signature.attr("class") = "Signature";

  return signature;
  END_RCPP
}


RcppExport SEXP FrameNum(SEXP sX) {
  NumericMatrix blockNum(sX);
  List predBlock = List::create(
        _["blockNum"] = blockNum,
        _["blockNumSparse"] = List(), // For now.
        _["blockFacSparse"] = R_NilValue, // For now.
        _["nPredNum"] = blockNum.ncol(),
        _["blockFac"] = IntegerMatrix(0),
        _["nPredFac"] = 0,
        _["nRow"] = blockNum.nrow(),
        _["facCard"] = IntegerVector(0),
        _["signature"] = move(FramemapBridge::wrapSignature(
                                        seq_len(blockNum.ncol()) - 1,
                                        List::create(0),
                                        colnames(blockNum),
                                        rownames(blockNum)))
                                );
  predBlock.attr("class") = "PredBlock";

  return predBlock;
}

// TODO:  Move column and row names to signature.
/**
   @brief Reads an S4 object containing (sparse) dgCMatrix.
 */
RcppExport SEXP FrameSparse(SEXP sX) {
  BEGIN_RCPP
  S4 spNum(sX);

  IntegerVector i;
  if (R_has_slot(sX, Rf_mkString("i"))) {
    i = spNum.slot("i");
  }
  IntegerVector j;
  if (R_has_slot(sX, Rf_mkString("j"))) {
    j = spNum.slot("j");
  }
  IntegerVector p;
  if (R_has_slot(sX, Rf_mkString("p"))) {
    p = spNum.slot("p");
  }

  if (!R_has_slot(sX, Rf_mkString("Dim"))) {
    stop("Expecting dimension slot");
  }
  if (!R_has_slot(sX, Rf_mkString("x"))) {
    stop("Pattern matrix:  NYI");
  }

  IntegerVector dim = spNum.slot("Dim");
  unsigned int nRow = dim[0];
  unsigned int nPred = dim[1];
  unique_ptr<BSCresc> bsCresc = make_unique<BSCresc>(nRow, nPred);

  // Divines the encoding format and packs appropriately.
  //
  if (i.length() == 0) {
    stop("Sparse form j/p:  NYI");
  }
  else if (p.length() == 0) {
    stop("Sparse form i/j:  NYI");
  }
  else if (j.length() == 0) {
    bsCresc->nzRow(&as<NumericVector>(spNum.slot("x"))[0], &i[0], &p[0]);
  }
  else {
    stop("Indeterminate sparse matrix format");
  }

  List blockNumSparse = List::create(
                                     _["valNum"] = bsCresc->getValNum(),
                                     _["rowStart"] = bsCresc->getRowStart(),
                                     _["runLength"] = bsCresc->getRunLength(),
                                     _["predStart"] = bsCresc->getPredStart());
  blockNumSparse.attr("class") = "BlockNumSparse";

  List dimNames;
  CharacterVector rowName = CharacterVector(0);
  CharacterVector colName = CharacterVector(0);
  if (R_has_slot(sX, Rf_mkString("Dimnames"))) {
    dimNames = spNum.slot("Dimnames");
    if (!Rf_isNull(dimNames[0])) {
      rowName = dimNames[0];
    }
    if (!Rf_isNull(dimNames[1])) {
      colName = dimNames[1];
    }
  }

  IntegerVector facCard(0);
  List predBlock = List::create(
        _["blockNum"] = NumericMatrix(0),
        _["nPredNum"] = nPred,
        _["blockNumSparse"] = move(blockNumSparse),
        _["blockFacSparse"] = R_NilValue, // For now.
        _["blockFac"] = IntegerMatrix(0),
        _["nPredFac"] = 0,
        _["nRow"] = nRow,
        _["facCard"] = facCard,
        _["signature"] = move(FramemapBridge::wrapSignature(seq_len(nPred) - 1,
                                        List::create(0),
                                        colName,
                                        rowName))
                                );

  predBlock.attr("class") = "PredBlock";

  return predBlock;
  END_RCPP
}


/**
   @brief Unwraps field values useful for prediction.
 */
List FramemapBridge::unwrapSignature(const List& sPredBlock) {
  BEGIN_RCPP
  checkPredblock(sPredBlock);
  return checkSignature(sPredBlock);
  END_RCPP
}


SEXP FramemapBridge::checkPredblock(const List &predBlock) {
  BEGIN_RCPP
  if (!predBlock.inherits("PredBlock")) {
    stop("Expecting PredBlock");
  }

  if (!Rf_isNull(predBlock["blockFacSparse"])) {
    stop ("Sparse factors:  NYI");
  }
  END_RCPP
}

void FramemapBridge::signatureUnwrap(const List& sTrain, IntegerVector &predMap, List &level) {
  List sSignature = checkSignature(sTrain);

  predMap = as<IntegerVector>((SEXP) sSignature["predMap"]);
  level = as<List>(sSignature["level"]);
}


SEXP FramemapBridge::checkSignature(const List &sParent) {
  BEGIN_RCPP
  List signature((SEXP) sParent["signature"]);
  if (!signature.inherits("Signature")) {
    stop("Expecting Signature");
  }

  return signature;
  END_RCPP
}


unique_ptr<FrameTrain> FramemapBridge::factoryTrain(
                    const vector<unsigned int> &facCard,
                    unsigned int nPred,
                    unsigned int nRow) {
  return make_unique<FrameTrain>(facCard, nPred, nRow);
}


unique_ptr<FramePredictBridge> FramemapBridge::factoryPredict(const List& sPredBlock) {
  checkPredblock(sPredBlock);
  return make_unique<FramePredictBridge>(
                 BlockNumBridge::Factory(sPredBlock),
                 BlockFacBridge::Factory(sPredBlock),
                 as<unsigned int>(sPredBlock["nRow"]));
}


FramePredictBridge::FramePredictBridge(
               unique_ptr<BlockNumBridge> blockNum_,
               unique_ptr<BlockFacBridge> blockFac_,
               unsigned int nRow_) :
  blockNum(move(blockNum_)),
  blockFac(move(blockFac_)),
  nRow(nRow_),
  framePredict(make_unique<FramePredict>(blockNum->getNum(),
                                         blockFac->getFac(),
                                         nRow)) {
}
