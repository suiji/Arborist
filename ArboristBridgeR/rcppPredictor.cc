// Copyright (C)  2012-2015   Mark Seligman
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
   @file rcppPredictor.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
 */

#include <R.h>
#include <Rcpp.h>

// Testing only:
//#include <iostream>

#include "predictor.h"

using namespace Rcpp;
using namespace std;

/**
  @brief Extracts contiguous factor and numeric blocks of observations from an R DataFrame by copying.  This can be quite slow for large predictor counts, for which DataTable may be a more suitable alternative.  Assumes columns are either factor or numeric.

  @param sX is the raw data frame.

  @param sNRow is the number of rows.

  @param sNCol is the number of columns.

  @param sFacCol is the number of factor-valued columns.

  @param sNumCol is the number of numeric-valued columns.

  @param sLevel is a vector of level counts for each column.

  @return Wrapped zero.
*/
RcppExport SEXP RcppPredictorFrame(SEXP sX, SEXP sNumCol, SEXP sFacCol, SEXP sLevels) {
  DataFrame xf(sX);
  IntegerVector levels(sLevels);
  int nRow = xf.nrows();
  int nCol = xf.length();
  int nColFac = as<int>(sFacCol);
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
    Predictor::FactorBlock(xFac.begin(), nColFac, facLevel.begin());
  }

  int nColNum = as<int>(sNumCol);
  if (nColNum > 0) {
    NumericMatrix xNum(nRow, nColNum);
    int colIdx = 0;
    for (int i = 0; i < nCol; i++) {
      if (levels[i] == 0) {
	xNum(_, colIdx++) = as<NumericVector>(xf[i]);
      }
    }
    Predictor::NumericBlock(xNum.begin(), nColNum);
  }

  return wrap(0);
}


/**
   @brief Caches a block of factor-valued predictors.

   @parameter sX is a contiguous block of factor-valued observations.

   @parameter sFacLevel is a vector of level counts for each column.

   @return Wrapped zero.
 */
RcppExport SEXP RcppPredictorFac(SEXP sX, SEXP sFacLevel) {
  IntegerMatrix xi(sX);
  IntegerVector facLevel(sFacLevel);

  Predictor::FactorBlock(xi.begin(), xi.ncol(), facLevel.begin());

  return wrap(0);
}


/**
   @brief Caches a block of numeric predictors.

   @parameter sX is a contiguous block of numeric observations.

   @parameter doClone indicates whether the block must be copied before over-writing.

   @return Wrapped zero.
 */
RcppExport SEXP RcppPredictorNum(SEXP sX, bool doClone) {
  NumericMatrix xn(sX);

  Predictor::NumericBlock(xn.begin(), xn.ncol(), doClone);

  return wrap(0);
}

/**
   @brief Caches a block of integer-valued predictors.

   @parameter sX is a contiguous block of integer-valued observations.

   @return Wrapped zero.
 */

RcppExport SEXP RcppPredictorInt(SEXP sX) {
  IntegerMatrix xi(sX);

  Predictor::IntegerBlock(xi.begin(), xi.ncol(), true);
  return wrap(0);
}


/**
   @brief Lights off the initializations used by the Predictor class.

   @param sPredProb is a vector of probabilities for selecting a given predictor as a splitting candidate.

   @param sNPred is the number of predictors being modelled.

   @param sNRow is the row count of the observation set.

   @return Wrapped zero.
 */
RcppExport SEXP RcppPredictorFactory(SEXP sPredProb, SEXP sNPred, SEXP sNRow) {
  if (Rf_isNull(sPredProb)) {
    Predictor::Factory(0, as<int>(sNPred), as<int>(sNRow));
  }
  else {
    NumericVector predProb(sPredProb);
    Predictor::Factory(predProb.begin(), as<int>(sNPred), as<int>(sNRow));
  }

  return wrap(0);
}


/**
   @return Exit status of block-integrity check.
 */
RcppExport SEXP RcppPredictorBlockEnd() {
  return wrap(Predictor::BlockEnd());
}
