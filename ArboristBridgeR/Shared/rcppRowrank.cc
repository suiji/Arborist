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
  std::vector<unsigned int> numOff(nPredNum);
  std::vector<double> numVal;
  if (nPredNum > 0) {
    if (!Rf_isNull(predBlock["blockNumRLE"])) {
      List blockNumRLE((SEXP) predBlock["blockNumRLE"]);
      if (!blockNumRLE.inherits("BlockNumRLE"))
	stop("Expecting BlockNumRLE");
      RowRank::PreSortNumRLE(NumericVector((SEXP) blockNumRLE["valNum"]).begin(), (unsigned int *) IntegerVector((SEXP) blockNumRLE["rowStart"]).begin(), (unsigned int*) IntegerVector((SEXP) blockNumRLE["runLength"]).begin(), nPredNum, nRow, row, rank, runLength, numOff, numVal);
    }
    else {
      NumericMatrix blockNum = predBlock["blockNum"];
      RowRank::PreSortNum(blockNum.begin(), nPredNum, nRow, row, rank, runLength, numOff, numVal);
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
      _["numOff"] = numOff,
      _["numVal"] = numVal
    );
  rowRank.attr("class") = "RowRank";

  return rowRank;
}


IntegerVector RcppRowrank::iv1 = IntegerVector(0);
IntegerVector RcppRowrank::iv2 = IntegerVector(0);
IntegerVector RcppRowrank::iv3 = IntegerVector(0);
IntegerVector RcppRowrank::iv4 = IntegerVector(0);
NumericVector RcppRowrank::nv1 = NumericVector(0);

void RcppRowrank::Unwrap(SEXP sRowRank, unsigned int *&feNumOff, double *&feNumVal, unsigned int *&feRow, unsigned int *&feRank, unsigned int *&feRLE, unsigned int &rleLength) {
  List rowRank(sRowRank);
  if (!rowRank.inherits("RowRank"))
    stop("Expecting RowRank");

  iv1 = IntegerVector((SEXP) rowRank["numOff"]);
  feNumOff = (unsigned int*) &iv1[0];

  nv1 = NumericVector((SEXP) rowRank["numVal"]);
  feNumVal = (double *) &nv1[0];
  
  iv2 = IntegerVector((SEXP) rowRank["row"]);
  feRow = (unsigned int *) &iv2[0];

  iv3 = IntegerVector((SEXP) rowRank["rank"]);
  feRank = (unsigned int *) &iv3[0];

  iv4 = IntegerVector((SEXP) rowRank["runLength"]);
  feRLE = (unsigned int *) &iv4[0];
  rleLength = iv4.length();
}


void RcppRowrank::Clear() {
  iv1 = IntegerVector(0);
  iv2 = IntegerVector(0);
  iv3 = IntegerVector(0);
  iv4 = IntegerVector(0);
  nv1 = NumericVector(0);
}
