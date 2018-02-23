// Copyright (C)  2012-2018   Mark Seligman
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
   @file rowrankBridge.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/


#include "rowrankBridge.h"

#include "coproc.h"
#include "frameblock.h"


/**
   @brief Builds row/rank maps as parallel arrays.

   @param sPredBlock is an (S3) PredBlock object.

   @return parallel row and rank arrays and the inverse numeric mapping.
 */
RcppExport SEXP Presort(SEXP sPredBlock) {
  BEGIN_RCPP
  List predBlock(sPredBlock);
  if (!predBlock.inherits("PredBlock")) {
    stop("Expecting PredBlock");
  }

  return RowRankBridge::Presort(predBlock);
  END_RCPP
}


List RowRankBridge::Presort(List &predBlock) {
  BEGIN_RCPP
  unsigned int nRow = as<unsigned int>(predBlock["nRow"]);
  unsigned int nPredNum = as<unsigned int>(predBlock["nPredNum"]);
  unsigned int nPredFac = as<unsigned int>(predBlock["nPredFac"]);

  vector<unsigned int> rank;
  vector<unsigned int> row;
  vector<unsigned int> runLength;
  vector<unsigned int> numOff(nPredNum);
  vector<double> numVal;

  if (!Rf_isNull(predBlock["blockNumSparse"])) {
    List blockNumSparse((SEXP) predBlock["blockNumSparse"]);
    if (!blockNumSparse.inherits("BlockNumSparse")) {
      stop("Expecting BlockNumSparse");
    }

    PreSortNumRLE(NumericVector((SEXP) blockNumSparse["valNum"]).begin(), (unsigned int *) IntegerVector((SEXP) blockNumSparse["rowStart"]).begin(), (unsigned int*) IntegerVector((SEXP) blockNumSparse["runLength"]).begin(), nPredNum, nRow, row, rank, runLength, numOff, numVal);
  }
  else {
    PreSortNum(NumericMatrix((SEXP) predBlock["blockNum"]).begin(), nPredNum, nRow, row, rank, runLength, numOff, numVal);
  }
  PreSortFac((unsigned int*) IntegerMatrix((SEXP) predBlock["blockFac"]).begin(), nPredFac, nRow, row, rank, runLength);

  List rowRank = List::create(
      _["row"] = row,			      
      _["rank"] = rank,
      _["runLength"] = runLength,
      _["numOff"] = numOff,
      _["numVal"] = numVal
    );
  rowRank.attr("class") = "RowRank";

  return rowRank;
  END_RCPP
}


RowRankBridge *RowRankBridge::Unwrap(SEXP sRowRank,
				     double autoCompress,
				     const Coproc *coproc,
				     const FrameTrain *frameTrain) {
  List rowRank = Legal(sRowRank);
  return new RowRankBridge(coproc,
			   frameTrain,
			   IntegerVector((SEXP) rowRank["numOff"]),
			   NumericVector((SEXP) rowRank["numVal"]),
			   IntegerVector((SEXP) rowRank["row"]),
			   IntegerVector((SEXP) rowRank["rank"]),
			   IntegerVector((SEXP) rowRank["runLength"]),
			   autoCompress);
}


List RowRankBridge::Legal(SEXP sRowRank) {
  BEGIN_RCPP
    List rowRank(sRowRank);
    if (!rowRank.inherits("RowRank")) {
      stop("Expecting RowRank");
    }
    return rowRank;

    END_RCPP
}


RowRankBridge::RowRankBridge(const Coproc *coproc,
			     const FrameTrain *frameTrain,
			     const IntegerVector &_numOff,
			     const NumericVector &_numVal,
			     const IntegerVector &_row,
			     const IntegerVector &_rank,
			     const IntegerVector &_runLength,
			     double _autoCompress) :
  RowRank(frameTrain,
	  (unsigned int*) &_row[0],
	  (unsigned int*) &_rank[0],
	  (unsigned int*) &_numOff[0],
	  (double *) &_numVal[0],
	  (unsigned int*) &_runLength[0],
	  _runLength.length(),
	  _autoCompress),
  numOff(_numOff),
  row(_row),
  rank(_rank),
  runLength(_runLength),
  numVal(_numVal) {
}

