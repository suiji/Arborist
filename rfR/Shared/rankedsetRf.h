// Copyright (C)  2012-2019  Mark Seligman
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
   @file rankedSetRf.h

   @brief C++ class definitions for managing RowRank object.

   @author Mark Seligman

 */


#ifndef ARBORIST_RANKEDSET_RF_H
#define ARBORIST_RANDKEDSET_RF_H

#include "rankedset.h"

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>


/**
   @brief External entry to presorting RowRank builder.

   @param sPredBlock is an R-style List containing frame block.

   @return R-style bundle representing row/rank structure.
 */
RcppExport SEXP Presort(SEXP sPredBlock);

/**
   @brief Rf specialization of BlockRanked (q.v.) caching pinned
   front-end containers.
 */
class BlockRankedRf final : public BlockRanked {
  const NumericVector numVal; // pinned
  const IntegerVector numOff; // pinned;

 public:
  BlockRankedRf(const NumericVector& numVal_,
                    const IntegerVector& numOff_);

  /**
     @brief Unwraps a sparse numerical block.
  */
  static unique_ptr<BlockRankedRf> unwrap(SEXP sBlockNum);
};

/**
   @brief Rf specialization of Core RowRank (q.v.) caching pinned
   front-end containeers.
 */
class RowRankRf : public RowRank {
  const IntegerVector row; // Pinned.
  const IntegerVector rank; // Pinned.
  const IntegerVector runLength; // Pinned.

 public:
  RowRankRf(const class Coproc *coproc,
                const class FrameTrain *frameTrain,
                const IntegerVector& row_,
                const IntegerVector& rank_,
                const IntegerVector& runLength_,
                double _autoCompress);

  /**
     @brief Checks that front end provides valid representation of a RowRank.

     @return List object containing valid representation.
   */
  static List checkRowRank(SEXP sRowRank);

  
  /**
     @brief Instantiates bridge-specialized RowRank from front end.
   */
  static unique_ptr<RowRankRf> unwrap(SEXP sRowRank,
                                          double autoCompress,
                                          const class Coproc* coproc,
                                          const class FrameTrain* frameTrain);
};


/**
   @brief Rf-level container caching
 */
class RankedSetRf {
  unique_ptr<RowRankRf> rowRank;
  unique_ptr<BlockRankedRf> numRanked;
  unique_ptr<RankedSet> rankedPair;
  
 public:
  /**
     @brief Static entry to block sorting.

     @param predBlock summarizes the predictor blocking scheme.

     @return R-style list of sorting summaries.
   */
  static List presort(List &predBlock);


  RankedSetRf(unique_ptr<RowRankRf> _rowRank,
                  unique_ptr<BlockRankedRf> _numRanked);

  /**
     @brief Getter for pointer to core pair object.
   */
  RankedSet *getPair() {
    return rankedPair.get();
  }

  
  /**
     @brief Unwraps an R-style representation of a RankedSet.
   */
  static unique_ptr<RankedSetRf> unwrap(SEXP sRowRank,
                                           double autoCompress,
                                           const class Coproc* coproc,
                                           const class FrameTrain* frameTrain);
};

#endif
