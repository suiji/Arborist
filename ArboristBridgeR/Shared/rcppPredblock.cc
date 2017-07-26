// Copyright (C)  2012-2017   Mark Seligman
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
   @file rcppPredBlock.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

// Testing only:
//#include <iostream>
//using namespace std;

#include "rcppPredblock.h"
#include "rowrank.h"


/**
  @brief Extracts contents of a data frame into numeric and (zero-based) factor blocks.  Can be quite slow for large predictor counts, as a linked list is being walked.

  @param sX is the raw data frame, with columns assumed to be either factor or numeric.

  @param sNRow is the number of rows.

  @param sNCol is the number of columns.

  @param sFacCol is the number of factor-valued columns.

  @param sNumCol is the number of numeric-valued columns.

  @param sLevel is a vector of level counts for each column.

  @return PredBlock with separate numeric and integer matrices.
*/
RcppExport SEXP RcppPredBlockFrame(SEXP sX, SEXP sNumElt, SEXP sFacElt, SEXP sLevels, SEXP sSigTrain) {
  BEGIN_RCPP
  DataFrame xf(sX);
  IntegerVector numElt = IntegerVector(sNumElt) - 1;
  IntegerVector facElt = IntegerVector(sFacElt) - 1;
  std::vector<unsigned int> levels = as<std::vector<unsigned int> >(sLevels);
  unsigned int nRow = xf.nrows();
  unsigned int nPredFac = facElt.length();
  unsigned int nPredNum = numElt.length();
  unsigned int nPred = nPredFac + nPredNum;

  IntegerVector predMap(nPred);
  IntegerVector facCard(0);
  IntegerMatrix xFac;
  NumericMatrix xNum;
  if (nPredNum > 0) {
    xNum = NumericMatrix(nRow, nPredNum);
  }
  else
    xNum = NumericMatrix(0, 0);
  if (nPredFac > 0) {
    facCard = IntegerVector(nPredFac);
    xFac = IntegerMatrix(nRow, nPredFac);
  }
  else {
    xFac = IntegerMatrix(0);
  }

  int numIdx = 0;
  int facIdx = 0;
  List level(nPredFac);
  for (unsigned int feIdx = 0; feIdx < nPred; feIdx++) {
    unsigned int card = levels[feIdx];
    if (card == 0) {
      xNum(_, numIdx) = as<NumericVector>(xf[feIdx]);
      predMap[numIdx++] = feIdx;
    }
    else {
      facCard[facIdx] = card;
      level[facIdx] = as<CharacterVector>(as<IntegerVector>(xf[feIdx]).attr("levels"));
      xFac(_, facIdx) = as<IntegerVector>(xf[feIdx]) - 1;
      predMap[nPredNum + facIdx++] = feIdx;
    }
  }

  // Factor positions must match those from training and values must conform.
  //
  if (!Rf_isNull(sSigTrain) && nPredFac > 0) {
    List sigTrain(sSigTrain);
    IntegerVector predTrain(as<IntegerVector>(sigTrain["predMap"]));
    if (!is_true(all(predMap == predTrain))) {
      stop("Training, prediction data types do not match");
    }

    List levelTrain(as<List>(sigTrain["level"]));
    RcppPredblock::FactorRemap(xFac, level, levelTrain);
  }
  List signature = List::create(
        _["predMap"] = predMap,
        _["level"] = level
	);
  signature.attr("class") = "Signature";
  
  List predBlock = List::create(
      _["colNames"] = colnames(xf),
      _["rowNames"] = rownames(xf),
      _["blockNum"] = xNum,
      _["nPredNum"] = nPredNum,
      _["blockNumRLE"] = R_NilValue, // For now.
      _["blockFacRLE"] = R_NilValue, // For now.
      _["blockFac"] = xFac,
      _["nPredFac"] = nPredFac,
      _["nRow"] = nRow,
      _["facCard"] = facCard,
      _["signature"] = signature
      );
  predBlock.attr("class") = "PredBlock";

  return predBlock;
  END_RCPP
}


void RcppPredblock::FactorRemap(IntegerMatrix &xFac, List &levelTest, List &levelTrain) {
  for (int col = 0; col < xFac.ncol(); col++) {
    CharacterVector colTest(as<CharacterVector>(levelTest[col]));
    CharacterVector colTrain(as<CharacterVector>(levelTrain[col]));
    if (is_true(any(colTest != colTrain))) {
      IntegerVector colMatch = match(colTest, colTrain);
      IntegerVector sq = seq(0, colTest.length() - 1);
      IntegerVector idxNonMatch = sq[is_na(colMatch)];
      if (idxNonMatch.length() > 0) {
	warning("Factor levels not observed in training:  employing proxy");
	int proxy = colTrain.length() + 1;
	colMatch[idxNonMatch] = proxy;
      }

      colMatch = colMatch - 1;  // match() is one-based.
      IntegerMatrix::Column xCol = xFac(_, col);
      IntegerVector colT(xCol);
      IntegerVector colRemap = colMatch[colT];
      xFac(_, col) = colRemap;
    }
  }
}


RcppExport SEXP RcppPredBlockNum(SEXP sX) {
  NumericMatrix blockNum(as<NumericMatrix>(sX));
  int nPred = blockNum.ncol();
  List dimnames = blockNum.attr("dimnames");
  List signature = List::create(
      _["predMap"] = seq_len(nPred) - 1,
      _["level"] = List::create(0)
  );
  signature.attr("class") = "Signature";

  IntegerVector facCard(0);
  List predBlock = List::create(
	_["colNames"] = colnames(blockNum),
	_["rowNames"] = rownames(blockNum),
	_["blockNum"] = blockNum,
	_["blockNumRLE"] = R_NilValue, // For now.
	_["blockFacRLE"] = R_NilValue, // For now.
	_["nPredNum"] = nPred,
        _["blockFac"] = IntegerMatrix(0),
	_["nPredFac"] = 0,
	_["nRow"] = blockNum.nrow(),
        _["facCard"] = facCard,
	_["signature"] = signature
      );
  predBlock.attr("class") = "PredBlock";

  return predBlock;
}


/**
   @brief Reads an S4 object containing (sparse) dgCMatrix.
 */
RcppExport SEXP RcppPredBlockSparse(SEXP sX) {
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


  IntegerVector dim = spNum.slot("Dim");
  unsigned int nRow = dim[0];
  unsigned int nPred = dim[1];

  // 'eltsNZ' holds the nonzero elements.
  NumericVector eltsNZ;

  if (R_has_slot(sX, Rf_mkString("x"))) {
    eltsNZ = spNum.slot("x");
  }
  else {
    stop("Pattern matrix:  NYI");
  }

  std::vector<double> valNum;
  std::vector<unsigned int> rowStart;
  std::vector<unsigned int> runLength;
  std::vector<unsigned int> predStart;

  // Divines the encoding format and packs appropriately.
  //
  if (i.length() == 0) {
    RcppPredblock::SparseJP(eltsNZ, j, p, nRow, valNum, rowStart, runLength);
  }
  else if (j.length() == 0) {
    RcppPredblock::SparseIP(eltsNZ, i, p, nRow, nPred, valNum, rowStart, runLength, predStart);
  }
  else if (p.length() == 0) {
    RcppPredblock::SparseIJ(eltsNZ, i, j, nRow, valNum, rowStart, runLength);
  }
  else {
    stop("Indeterminate sparse matrix format");
  }

  List blockNumRLE = List::create(
	  _["valNum"] = valNum,
	  _["rowStart"] = rowStart,
	  _["runLength"] = runLength,
	  _["predStart"] = predStart);
  blockNumRLE.attr("class") = "BlockNumRLE";

  List dimNames;
  CharacterVector rowName, colName;
  if (R_has_slot(sX, Rf_mkString("Dimnames"))) {
    dimNames = spNum.slot("Dimnames");
    if (!Rf_isNull(dimNames[0])) {
      rowName = dimNames[0];
    }
    if (!Rf_isNull(dimNames[1])) {
      colName = dimNames[1];
    }
  }

  List signature = List::create(
      _["predMap"] = seq_len(nPred) - 1,
      _["level"] = List::create(0)
  );
  signature.attr("class") = "Signature";
  IntegerVector facCard(0);

  List predBlock = List::create(
	_["colNames"] = colName,
	_["rowNames"] = rowName,
	_["blockNum"] = NumericMatrix(0),
	_["nPredNum"] = nPred,
	_["blockNumRLE"] = blockNumRLE,
	_["blockFacRLE"] = R_NilValue, // For now.
        _["blockFac"] = IntegerMatrix(0),
	_["nPredFac"] = 0,
	_["nRow"] = nRow,
        _["facCard"] = facCard,
	_["signature"] = signature
      );

  predBlock.attr("class") = "PredBlock";

  return predBlock;
  END_RCPP
}


// 'i' in [0, nRow-1] list rows with nonzero elements.
// 'p' holds the starting offset for each column in 'eltsNZ'.
//    Repeated values indicate full-zero columns. 
//
void RcppPredblock::SparseIP(const NumericVector &eltsNZ, const IntegerVector &i, const IntegerVector &p, unsigned int nRow, unsigned int nCol, std::vector<double> &valNum, std::vector<unsigned int> &rowStart, std::vector<unsigned int> &runLength, std::vector<unsigned int> &predStart) {
  // Pre-scans column heights. 'p' has length one greater than number
  // of columns, providing ready access to heights.
  const double zero = 0.0;
  std::vector<unsigned int> nzHeight(p.length());
  unsigned int idxStart = p[0];
  for (R_len_t colIdx = 1; colIdx < p.length(); colIdx++) {
    nzHeight[colIdx - 1] = p[colIdx] - idxStart;
    idxStart = p[colIdx];
  }
  
  for (unsigned int colIdx = 0; colIdx < nCol; colIdx++) {
    unsigned int height = nzHeight[colIdx];
    predStart.push_back(valNum.size());
    if (height == 0) {
      valNum.push_back(zero);
      runLength.push_back(nRow);
      rowStart.push_back(0);
    }
    else {
      unsigned int nzPrev = nRow; // Inattainable row value.
      // Row indices into 'i' and 'x' are zero-based.
      unsigned int idxStart = p[colIdx];
      unsigned int idxEnd = idxStart + height;
      for (unsigned int rowIdx = idxStart; rowIdx < idxEnd; rowIdx++) {
        unsigned int nzRow = i[rowIdx];
        if (nzPrev == nRow && nzRow > 0) { // Zeroes lead.
	  valNum.push_back(zero);
	  runLength.push_back(nzRow);
	  rowStart.push_back(0);
	}
	else if (nzRow > nzPrev + 1) { // Zeroes precede.
	  valNum.push_back(zero);
	  runLength.push_back(nzRow - (nzPrev + 1));
	  rowStart.push_back(nzPrev + 1);
	}
	valNum.push_back(eltsNZ[rowIdx]);
	runLength.push_back(1);
	rowStart.push_back(nzRow);
	nzPrev = nzRow;
      }
      if (nzPrev + 1 < nRow) { // Zeroes trail.
	valNum.push_back(zero);
	runLength.push_back(nRow - nzPrev - 1);
	rowStart.push_back(nzPrev + 1);
      }
    }
  }
}



void RcppPredblock::SparseJP(NumericVector &eltsNZ, IntegerVector &j, IntegerVector &p, unsigned int nRow, std::vector<double> &valNum, std::vector<unsigned int> &rowStart, std::vector<unsigned int> &runLength) {
  try {
    throw std::domain_error("Sparse form j/p:  NYI");
  }
  catch (std::exception &ex) {
    forward_exception_to_r(ex);
  }
}


    // 'i' holds row indices of nonzero elements.
    // 'j' " column " "
void RcppPredblock::SparseIJ(NumericVector &eltsNZ, IntegerVector &i, IntegerVector &j, unsigned int nRow, std::vector<double> &valNum, std::vector<unsigned int> &rowStart, std::vector<unsigned int> &runLength) {
  try {
    throw std::domain_error("Sparse form i/j:  NYI");
  }
  catch (std::exception &ex) {
    forward_exception_to_r(ex);
  }
}


/**
   @brief Unwraps field values useful for prediction.
 */
void RcppPredblock::Unwrap(SEXP sPredBlock, unsigned int &_nRow, unsigned int &_nPredNum, unsigned int &_nPredFac, NumericMatrix &_blockNum, IntegerMatrix &_blockFac, std::vector<double> &_valNum, std::vector<unsigned int> &_rowStart, std::vector<unsigned int> &_runLength, std::vector<unsigned int> &_predStart) {
  List predBlock(sPredBlock);

  try {
    if (!predBlock.inherits("PredBlock")) {
      throw std::domain_error("Expecting PredBlock");
    }
  }
  catch(std::exception &ex) {
    forward_exception_to_r(ex);
  }
  
  _nRow = as<unsigned int>((SEXP) predBlock["nRow"]);
  _nPredFac = as<unsigned int>((SEXP) predBlock["nPredFac"]);
  _nPredNum = as<unsigned int>((SEXP) predBlock["nPredNum"]);
  if (!Rf_isNull(predBlock["blockNumRLE"])) {
    List blockNumRLE((SEXP) predBlock["blockNumRLE"]);
    _valNum = as<std::vector<double> >((SEXP) blockNumRLE["valNum"]);
    _rowStart = as<std::vector<unsigned int> >((SEXP) blockNumRLE["rowStart"]);
    _runLength = as<std::vector<unsigned int> >((SEXP) blockNumRLE["runLength"]);
    _predStart = as<std::vector<unsigned int> >((SEXP) blockNumRLE["predStart"]);
  }
  else {
    _blockNum = as<NumericMatrix>((SEXP) predBlock["blockNum"]);
  }

  try {
    if (!Rf_isNull(predBlock["blockFacRLE"])) {
      throw std::domain_error("Sparse factors:  NYI");
    }
  }
  catch(std::exception &ex) {
    forward_exception_to_r(ex);
  }

  _blockFac = as<IntegerMatrix>((SEXP) predBlock["blockFac"]);
}


/**
   @brief Unwraps field values useful for export.
 */
void RcppPredblock::SignatureUnwrap(SEXP sSignature, IntegerVector &_predMap, List &_level) {
  List signature(sSignature);
  try {
    if (!signature.inherits("Signature")) {
      throw std::domain_error("Expecting Signature");
    }
  }
  catch(std::exception &ex) {
    forward_exception_to_r(ex);
  }

  _predMap = as<IntegerVector>((SEXP) signature["predMap"]);
  _level = as<List>(signature["level"]);
}
