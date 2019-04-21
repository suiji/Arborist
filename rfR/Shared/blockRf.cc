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
   @file blockRf.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "blockRf.h"


BlockFacRf::BlockFacRf(const IntegerMatrix &fac) :
  facT(transpose(fac)) {
  blockFac = make_unique<BlockFac>((unsigned int*)facT.begin(), fac.ncol());
}


/**
  @brief Sparse constructor.
 */
BlockNumSparseRf::BlockNumSparseRf(const NumericVector &_val,
				     const IntegerVector &_rowStart,
				     const IntegerVector &_runLength,
				     const IntegerVector &_predStart) :
  val(_val),
  rowStart(_rowStart),
  runLength(_runLength),
  predStart(_predStart) {
  blockNum = make_unique<BlockNumSparse>(val.begin(),
				      (unsigned int *) rowStart.begin(),
				      (unsigned int *) runLength.begin(),
				      (unsigned int *) predStart.begin(),
				      predStart.length());
}


/**
   @brief Dense constructor
 */
BlockNumDenseRf::BlockNumDenseRf(const NumericMatrix &num) {
  numT = transpose(num);
  blockNum = make_unique<BlockNumDense>(numT.begin(), num.ncol());
}


unique_ptr<BlockFacRf> BlockFacRf::Factory(const List &predBlock) {
  return make_unique<BlockFacRf>(IntegerMatrix((SEXP) predBlock["blockFac"]));
}

unique_ptr<BlockNumRf> BlockNumRf::Factory(const List &predBlock) {
  List blockNumSparse((SEXP) predBlock["blockNumSparse"]);
  if (blockNumSparse.length() > 0) {
    return make_unique<BlockNumSparseRf>(
		  NumericVector((SEXP) blockNumSparse["valNum"]),
		  IntegerVector((SEXP) blockNumSparse["rowStart"]),
		  IntegerVector((SEXP) blockNumSparse["runLength"]),
		  IntegerVector((SEXP) blockNumSparse["predStart"]));
  }
  else {
    return make_unique<BlockNumDenseRf>(
				 NumericMatrix((SEXP) predBlock["blockNum"])
					 );
  }
}
