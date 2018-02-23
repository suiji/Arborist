// Copyright (C)  2012-2018  Mark Seligman
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

#include "rowrank.h"

/**
   @brief External entry to presorting RowRank builder.

   @param R-style List containing frame block.

   @return R-style bundle representing row/rank structure.
 */
RcppExport SEXP Presort(SEXP sPredBlock);


/**
   @brief Bridge specialization of Core RowRank, q.v.
 */
class RowRankBridge : public RowRank {
  const IntegerVector numOff;
  const IntegerVector row;
  const IntegerVector rank;
  const IntegerVector runLength;
  const NumericVector numVal;

 public:
  RowRankBridge(const class Coproc *coproc,
		const class FrameTrain *frameTrain,
		const IntegerVector &_numOff,
		const NumericVector &_numVal,
		const IntegerVector &_row,
		const IntegerVector &_rank,
		const IntegerVector &_runLength,
		double _autoCompress);

  static List Legal(SEXP sRowRank);

  static List Presort(List &predBlock);
  
  static RowRankBridge *Unwrap(SEXP sRowRank,
			       double autoCompress,
			       const class Coproc *coproc,
			       const class FrameTrain *frameTrain);
};


#endif
