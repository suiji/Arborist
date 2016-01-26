// Copyright (C)  2012-2016   Mark Seligman
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

#include <R.h>
#include <Rcpp.h>

#include "rowrank.h"
// Testing only:
//#include <iostream>


using namespace Rcpp;
using namespace std;

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
RcppExport SEXP RcppPredBlockFrame(SEXP sX, SEXP sNumCol, SEXP sFacCol, SEXP sLevels) {
  DataFrame xf(sX);

  IntegerVector levels(sLevels);
  int nRow = xf.nrows();
  int nPredFac = as<int>(sFacCol);
  int nPredNum = as<int>(sNumCol);
  int nPred = nPredFac + nPredNum;
  IntegerVector predMap(nPred);

  IntegerVector facCard = 0;
  IntegerMatrix xFac = 0;
  NumericMatrix xNum = 0;
  if (nPredNum > 0) {
    xNum = NumericMatrix(nRow, nPredNum);
  }
  if (nPredFac > 0) {
    facCard = IntegerVector(nPredFac); // Compressed factor vector.
    xFac = IntegerMatrix(nRow, nPredFac);
  }

  int numIdx = 0;
  int facIdx = 0;
  for (int feIdx = 0; feIdx < nPred; feIdx++) {
    int card = levels[feIdx];
    if (card == 0) {
      predMap[numIdx] = feIdx;
      xNum(_, numIdx++) = as<NumericVector>(xf[feIdx]);
    }
    else {
      facCard[facIdx] = card;
      predMap[nPredNum + facIdx] = feIdx;
      xFac(_, facIdx++) = as<IntegerVector>(xf[feIdx]) - 1;
    }
  }

  List predBlock = List::create(
      _["colNames"] = colnames(xf),
      _["rowNames"] = rownames(xf),
      _["blockNum"] = xNum,
      _["nPredNum"] = nPredNum,
      _["blockFac"] = xFac,
      _["nPredFac"] = nPredFac,
      _["nRow"] = nRow,
      _["facCard"] = facCard,
      _["predMap"] = predMap
      );
  predBlock.attr("class") = "PredBlock";

  return predBlock;
}


RcppExport SEXP RcppPredBlockNum(SEXP sX) {
  NumericMatrix blockNum(as<NumericMatrix>(sX));
  int nPred = blockNum.ncol();
  List dimnames = blockNum.attr("dimnames");
  List predBlock = List::create(
	_["colNames"] = colnames(blockNum),
	_["rowNames"] = rownames(blockNum),
	_["blockNum"] = blockNum,
	_["nPredNum"] = nPred,
        _["blockFac"] = IntegerMatrix(0),
	_["nPredFac"] = 0,
	_["nRow"] = blockNum.nrow(),
        _["facCard"] = IntegerVector(0),
        _["predMap"] = seq_len(nPred)-1
      );
  predBlock.attr("class") = "PredBlock";

  return predBlock;
}

void RcppPredblockUnwrap(SEXP sPredBlock, int &_nRow, int &_nPredNum, int &_nPredFac, NumericMatrix &_blockNum, IntegerMatrix &_blockFac) {
  List predBlock(sPredBlock);
  if (!predBlock.inherits("PredBlock"))
    stop("Expecting PredBlock");
  
  _nRow = as<int>((SEXP) predBlock["nRow"]);
  _nPredFac = as<int>((SEXP) predBlock["nPredFac"]);
  _nPredNum = as<int>((SEXP) predBlock["nPredNum"]);
  _blockNum = as<NumericMatrix>((SEXP) predBlock["blockNum"]);
  _blockFac = as<IntegerMatrix>((SEXP) predBlock["blockFac"]);
}
