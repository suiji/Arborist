// Copyright (C)  2012-2019   Mark Seligman
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
   @file blockBridge.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/


#include "blockBridge.h"


BlockFacBridge::BlockFacBridge(const IntegerMatrix &fac) :
  facT(transpose(fac)) {
  blockFac = make_unique<BlockFac>((unsigned int*)facT.begin(), fac.ncol());
}


/**
  @brief Sparse constructor.
 */
BlockSparseBridge::BlockSparseBridge(const NumericVector &_val,
				     const IntegerVector &_rowStart,
				     const IntegerVector &_runLength,
				     const IntegerVector &_predStart) :
  val(_val),
  rowStart(_rowStart),
  runLength(_runLength),
  predStart(_predStart) {
  blockNum = make_unique<BlockSparse>(val.begin(),
				      (unsigned int *) rowStart.begin(),
				      (unsigned int *) runLength.begin(),
				      (unsigned int *) predStart.begin(),
				      predStart.length());
}


/**
   @brief Dense constructor
 */
BlockDenseBridge::BlockDenseBridge(const NumericMatrix &num) {
  numT = transpose(num);
  blockNum = make_unique<BlockNumDense>(numT.begin(), num.ncol());
}


unique_ptr<BlockFacBridge> BlockFacBridge::Factory(const List &predBlock) {
  return make_unique<BlockFacBridge>(IntegerMatrix((SEXP) predBlock["blockFac"]));
}

unique_ptr<BlockNumBridge> BlockNumBridge::Factory(const List &predBlock) {
  List blockNumSparse((SEXP) predBlock["blockNumSparse"]);
  if (blockNumSparse.length() > 0) {
    return make_unique<BlockSparseBridge>(
		  NumericVector((SEXP) blockNumSparse["valNum"]),
		  IntegerVector((SEXP) blockNumSparse["rowStart"]),
		  IntegerVector((SEXP) blockNumSparse["runLength"]),
		  IntegerVector((SEXP) blockNumSparse["predStart"]));
  }
  else {
    return make_unique<BlockDenseBridge>(
				 NumericMatrix((SEXP) predBlock["blockNum"])
					 );
  }
}
