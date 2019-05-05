// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of rfR.
//
// rfR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rfR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file rankedSetRf.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "rankedsetRf.h"

#include "coproc.h"
#include "framemapRf.h"


RowRankRf::RowRankRf(const Coproc *coproc,
                             const FrameMap *frameTrain,
                             const IntegerVector &_row,
                             const IntegerVector &_rank,
                             const IntegerVector &_runLength,
                             double _autoCompress) :
  RowRank(frameTrain,
          (unsigned int*) &_row[0],
          (unsigned int*) &_rank[0],
          (unsigned int*) &_runLength[0],
          _runLength.length(),
          _autoCompress),
  row(_row),
  rank(_rank),
  runLength(_runLength) {
}


BlockRankedRf::BlockRankedRf(const NumericVector &_numVal,
                                     const IntegerVector &_numOff) :
  BlockRanked(&_numVal[0], (unsigned int*) &_numOff[0]),
  numVal(_numVal),
  numOff(_numOff) {
}


RankedSetRf::RankedSetRf(unique_ptr<RowRankRf> _rowRank,
                                 unique_ptr<BlockRankedRf> _numRanked) :
  rowRank(move(_rowRank)),
  numRanked(move(_numRanked)) {
  rankedPair = make_unique<RankedSet>(rowRank.get(), numRanked.get());
}


RcppExport SEXP Presort(SEXP sPredBlock) {
  BEGIN_RCPP

  List predBlock(sPredBlock);
  if (!predBlock.inherits("PredBlock")) {
    stop("Expecting PredBlock");
  }
  return RankedSetRf::presort(predBlock);

  END_RCPP
}


List RankedSetRf::presort(List &predBlock) {
  BEGIN_RCPP

  auto rankedPre = make_unique<RankedPre>(as<unsigned int>(predBlock["nRow"]),
                                       as<unsigned int>(predBlock["nPredNum"]),
                                       as<unsigned int>(predBlock["nPredFac"])
                                       );
  List blockNumSparse((SEXP) predBlock["blockNumSparse"]);
  if (blockNumSparse.length() > 0) {
    if (!blockNumSparse.inherits("BlockNumSparse")) {
      stop("Expecting BlockNumSparse");
    }
    rankedPre->numSparse(NumericVector((SEXP) blockNumSparse["valNum"]).begin(),
     (unsigned int*) IntegerVector((SEXP) blockNumSparse["rowStart"]).begin(),
     (unsigned int*) IntegerVector((SEXP) blockNumSparse["runLength"]).begin()
                     );
  }
  else {
    rankedPre->numDense(NumericMatrix((SEXP) predBlock["blockNum"]).begin());
  }
  rankedPre->facDense((unsigned int*) IntegerMatrix((SEXP) predBlock["blockFac"]).begin());

  // Ranked numerical values for splitting-value interpolation.
  //
  List numRanked = List::create(
                                _["numVal"] = rankedPre->NumVal(),
                                _["numOff"] = rankedPre->NumOff()
                                );
  numRanked.attr("class") = "NumRanked";

  List rowRank = List::create(
                              _["row"] = rankedPre->Row(),
                              _["rank"] = rankedPre->Rank(),
                              _["runLength"] = rankedPre->RunLength()
                              );
  rowRank.attr("class") = "RowRank";

  List setOut = List::create(
                             _["rowRank"] = move(rowRank),
                             _["numRanked"] = move(numRanked)
                                 );
  setOut.attr("class") = "RankedSet";

  return setOut;

  END_RCPP
}


unique_ptr<RowRankRf> RowRankRf::unwrap(SEXP sRankedSet,
                                                double autoCompress,
                                                const Coproc *coproc,
                                                const FrameMap *frameTrain) {
  List rankedSet(sRankedSet);
  List rowRank = checkRowRank(rankedSet["rowRank"]);
  return make_unique<RowRankRf>(coproc,
                           frameTrain,
                           IntegerVector((SEXP) rowRank["row"]),
                           IntegerVector((SEXP) rowRank["rank"]),
                           IntegerVector((SEXP) rowRank["runLength"]),
                           autoCompress);
}


unique_ptr<BlockRankedRf> BlockRankedRf::unwrap(SEXP sRankedSet) {
  List rankedSet(sRankedSet);
  List blockNum((SEXP) rankedSet["numRanked"]);
  return make_unique<BlockRankedRf>(
                                NumericVector((SEXP) blockNum["numVal"]),
                                IntegerVector((SEXP) blockNum["numOff"]));
}


unique_ptr<RankedSetRf> RankedSetRf::unwrap(
                    SEXP sRankedSet,
                    double autoCompress,
                    const Coproc *coproc,
                    const FrameMap *frameTrain) {
  return make_unique<RankedSetRf>(
      RowRankRf::unwrap(sRankedSet,autoCompress, coproc, frameTrain),
      BlockRankedRf::unwrap(sRankedSet)
                                      );
}


List RowRankRf::checkRowRank(SEXP sRowRank) {
  BEGIN_RCPP

  List rowRank(sRowRank);
  if (!rowRank.inherits("RowRank")) {
    stop("Expecting RowRank");
  }
  return rowRank;

 END_RCPP
}
