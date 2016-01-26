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
   @file rcppRowrank.cc

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
   @brief Builds row/rank maps as parallel arrays.

   @param sPredBlock is an (S3) PredBlock object.

   @return parallel row and rank arrays and the inverse numeric mapping.
 */
RcppExport SEXP RcppRowRank(SEXP sPredBlock) {
  List predBlock(sPredBlock);
  if (!predBlock.inherits("PredBlock"))
    stop("Expecting PredBlock");

  unsigned int nRow = as<unsigned int>(predBlock["nRow"]);
  unsigned int nPredNum = as<unsigned int>(predBlock["nPredNum"]);
  unsigned int nPredFac = as<unsigned int>(predBlock["nPredFac"]);
  unsigned int nPred = nPredNum + nPredFac;
  IntegerVector rank = IntegerVector(nRow * nPred);
  IntegerVector row = IntegerVector(nRow * nPred);
  IntegerVector invNum = 0;
  if (nPredNum > 0) {
    invNum = IntegerVector(nRow * nPredNum);
    NumericMatrix blockNum(as<NumericMatrix>(predBlock["blockNum"]));
    RowRank::PreSortNum(blockNum.begin(), nPredNum, nRow, row.begin(), rank.begin(), invNum.begin());
  }
  if (nPredFac > 0) {
    IntegerMatrix blockFac(as<IntegerMatrix>(predBlock["blockFac"]));
    RowRank::PreSortFac(blockFac.begin(), nPredNum, nPredFac, nRow, row.begin(), rank.begin());
  }
  
  List rowRank = List::create(
      _["row"] = row,			      
      _["rank"] = rank,
      _["invNum"] = invNum
    );
  rowRank.attr("class") = "RowRank";

  return rowRank;
}
