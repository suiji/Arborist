# Copyright (C)  2012-2014   Mark Seligman
##
## This file is part of ArboristBridgeR.
##
## ArboristBridgeR is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## ArboristBridgeR is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.


#include <R.h>
#include <Rcpp.h>
#include <iostream>

#include "predictor.h"

using namespace Rcpp;
using namespace std;

// Assumes columns are either factor or numeric.
//
RcppExport SEXP RcppPredictorFrame(SEXP sx, SEXP sNRow, SEXP sNCol, SEXP sFacCol, SEXP sNumCol, SEXP sLevels) {
  DataFrame xf(sx);
  IntegerVector levels(sLevels);
  int nRow = as<int>(sNRow);
  int nCol = as<int>(sNCol);
  int nColNum = as<int>(sNumCol);
  int nColFac = as<int>(sFacCol);
  
  // ASSERTION:
  if (nColNum + nColFac != nCol)
    cout << "Unrecognized data frame layout." << endl;
    
  if (nColFac > 0) {
    IntegerVector facLevel(nColFac); // Compressed factor vector.
    IntegerMatrix xFac(nRow, nColFac);
    int colIdx = 0;
    for (int i = 0; i < nCol; i++) {
      if (levels[i] > 0) {
	facLevel[colIdx] = levels[i];
	xFac(_,colIdx++) = as<IntegerVector>(xf[i]);
      }
    }
    Predictor::FactorBlock(xFac.begin(), nRow, nColFac, facLevel.begin());
  }
  if (nColNum > 0) {
    NumericMatrix xNum(nRow, nColNum);
    int colIdx = 0;
    for (int i = 0; i < nCol; i++) {
      if (levels[i] == 0) {
	xNum(_, colIdx++) = as<NumericVector>(xf[i]);
      }
    }
    Predictor::NumericBlock(xNum.begin(), nRow, nColNum);
  }

  return wrap(0);
}

// Already duplicated from containing data frame.
//
RcppExport SEXP RcppPredictorFac(SEXP sx, SEXP sFacLevel) {
  IntegerMatrix xi(sx);
  IntegerVector facLevel(sFacLevel);

  Predictor::FactorBlock(xi.begin(), xi.nrow(), xi.ncol(), facLevel.begin());

  return wrap(0);
}
RcppExport SEXP RcppPredictorNum(SEXP sx, bool doClone) {
  NumericMatrix xn(sx);

  Predictor::NumericBlock(xn.begin(), xn.nrow(), xn.ncol(), doClone);
  return wrap(0);
}

// NYI
//
RcppExport SEXP RcppPredictorInt(SEXP sx) {
  IntegerMatrix xi(sx);

  Predictor::IntegerBlock(xi.begin(), xi.nrow(), xi.ncol(), true);
  return wrap(0);
}
