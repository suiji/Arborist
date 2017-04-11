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
   @file rcppRowrank.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/


#include "rcppRowrank.h"
#include "rowrank.h"

// Testing only:
//#include <iostream>
//using namespace std;


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
  std::vector<unsigned int> rank;
  std::vector<unsigned int> row;
  std::vector<unsigned int> runLength;

  std::vector<unsigned int> numOffset(nPredNum);
  std::vector<double> numVal;
  if (nPredNum > 0) {
    if (!Rf_isNull(predBlock["blockNumRLE"])) {
      List blockNumRLE((SEXP) predBlock["blockNumRLE"]);
      if (!blockNumRLE.inherits("BlockNumRLE"))
	stop("Expecting BlockNumRLE");
      const std::vector<double> &valNum = as<std::vector<double> >((SEXP) blockNumRLE["valNum"]);
      const std::vector<unsigned int> &rowStart = as<std::vector<unsigned int> >((SEXP) blockNumRLE["rowStart"]);
      const std::vector<unsigned int> &rLength = as<std::vector<unsigned int> >((SEXP) blockNumRLE["runLength"]);
      RowRank::PreSortNumRLE(valNum, rowStart, rLength, nPredNum, nRow, row, rank, runLength, numOffset, numVal);
    }
    else {
      NumericMatrix blockNum = predBlock["blockNum"];
      RowRank::PreSortNum(blockNum.begin(), nPredNum, nRow, row, rank, runLength, numOffset, numVal);
    }
  }

  if (nPredFac > 0) {
    IntegerMatrix blockFac = predBlock["blockFac"];
    RowRank::PreSortFac((unsigned int*) blockFac.begin(), nPredFac, nRow, row, rank, runLength);
  }

  List rowRank = List::create(
      _["row"] = row,			      
      _["rank"] = rank,
      _["runLength"] = runLength,
      _["numOff"] = numOffset,
      _["numVal"] = numVal
    );
  rowRank.attr("class") = "RowRank";

  return rowRank;
}


void RcppRowrank::Unwrap(SEXP sRowRank, unsigned int *&feNumOff, double *&feNumVal, unsigned int *&feRow, unsigned int *&feRank, unsigned int *&feRLE, unsigned int &rleLength) {
  List rowRank(sRowRank);
  if (!rowRank.inherits("RowRank"))
    stop("Expecting RowRank");

  IntegerVector t1((SEXP) rowRank["numOff"]);
  feNumOff = (unsigned int*) &t1[0];

  NumericVector t2((SEXP) rowRank["numVal"]);
  feNumVal = &t2[0];

  IntegerVector t3((SEXP) rowRank["row"]);
  feRow = (unsigned int *) &t3[0];

  IntegerVector t4((SEXP) rowRank["rank"]);
  feRank = (unsigned int*) &t4[0];

  IntegerVector t5((SEXP) rowRank["runLength"]);
  feRLE = (unsigned int*) &t5[0];

  rleLength = IntegerVector((SEXP) rowRank["runLength"]).length();
}
