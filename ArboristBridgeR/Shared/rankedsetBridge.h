// Copyright (C)  2012-2019  Mark Seligman
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
   @file rowrankBridge.h

   @brief C++ class definitions for managing RowRank object.

   @author Mark Seligman

 */


#ifndef ARBORIST_ROWRANK_BRIDGE_H
#define ARBORIST_ROWRANK_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>

#include "rowrank.h"

/**
   @brief External entry to presorting RowRank builder.

   @param sPredBlock is an R-style List containing frame block.

   @return R-style bundle representing row/rank structure.
 */
RcppExport SEXP Presort(SEXP sPredBlock);

/**
   @brief Bridge specialization of BlockRanked (q.v.) caching pinned
   front-end containers.
 */
class BlockRankedBridge final : public BlockRanked {
  const NumericVector numVal; // pinned
  const IntegerVector numOff; // pinned;

 public:
  BlockRankedBridge(const NumericVector &_numVal,
                    const IntegerVector &_numOff);

  /**
     @brief Unwraps a sparse numerical block.
  */
  static unique_ptr<BlockRankedBridge> unwrap(SEXP sBlockNum);
};

/**
   @brief Bridge specialization of Core RowRank (q.v.) caching pinned
   front-end containeers.
 */
class RowRankBridge : public RowRank {
  const IntegerVector row; // Pinned.
  const IntegerVector rank; // Pinned.
  const IntegerVector runLength; // Pinned.

 public:
  RowRankBridge(const class Coproc *coproc,
                const class FrameTrain *frameTrain,
                const IntegerVector &_row,
                const IntegerVector &_rank,
                const IntegerVector &_runLength,
                double _autoCompress);

  /**
     @brief Checks that front end provides valid representation of a RowRank.

     @return List object containing valid representation.
   */
  static List checkRowRank(SEXP sRowRank);

  
  /**
     @brief Instantiates bridge-specialized RowRank from front end.
   */
  static unique_ptr<RowRankBridge> unwrap(SEXP sRowRank,
                                          double autoCompress,
                                          const class Coproc *coproc,
                                          const class FrameTrain *frameTrain);
};


/**
   @brief Bridge-level container caching
 */
class RankedSetBridge {
  unique_ptr<RowRankBridge> rowRank;
  unique_ptr<BlockRankedBridge> numRanked;
  unique_ptr<RankedSet> rankedPair;
  
 public:
  /**
     @brief Static entry to block sorting.

     @param predBlock summarizes the predictor blocking scheme.

     @return R-style list of sorting summaries.
   */
  static List presort(List &predBlock);


  RankedSetBridge(unique_ptr<RowRankBridge> _rowRank,
                  unique_ptr<BlockRankedBridge> _numRanked);

  /**
     @brief Getter for pointer to core pair object.
   */
  RankedSet *getPair() {
    return rankedPair.get();
  }

  
  /**
     @brief Unwraps an R-style representation of a RankedSet.
   */
  static unique_ptr<RankedSetBridge> unwrap(SEXP sRowRank,
                                           double autoCompress,
                                           const class Coproc *coproc,
                                           const class FrameTrain *frameTrain);
};

#endif
