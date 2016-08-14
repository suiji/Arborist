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
  IntegerMatrix rank(nRow, nPred);
  IntegerMatrix row(nRow, nPred);
  IntegerMatrix invNum;
  if (nPredNum > 0) {
    invNum = IntegerMatrix(nRow, nPredNum);
    NumericMatrix blockNum = predBlock["blockNum"];
    RowRank::PreSortNum(blockNum.begin(), nPredNum, nRow, (unsigned int *) row.begin(), (unsigned int*) rank.begin(), (unsigned int *) invNum.begin());
  }
  else {
    invNum = IntegerMatrix(0,0);
  }

  if (nPredFac > 0) {
    unsigned int facStart = nPredNum * nRow;
    IntegerMatrix blockFac = predBlock["blockFac"];
    RowRank::PreSortFac((unsigned int*) blockFac.begin(), nPredFac, nRow, (unsigned int*) &row[facStart], (unsigned int*) &rank[facStart]);
  }
  
  List rowRank = List::create(
      _["row"] = row,			      
      _["rank"] = rank,
      _["invNum"] = invNum
    );
  rowRank.attr("class") = "RowRank";

  return rowRank;
}
