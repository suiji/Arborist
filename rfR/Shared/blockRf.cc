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
  blockFac = make_unique<BlockDense<unsigned int> >(fac.ncol(), (unsigned int*)facT.begin());
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
  blockNum = make_unique<BlockSparse<double> >(predStart.length(),
                                               val.begin(),
                              (const unsigned int *) rowStart.begin(),
			      (const unsigned int *) runLength.begin(),
                              (const unsigned int *) predStart.begin());
}


/**
   @brief Dense constructor
 */
BlockNumDenseRf::BlockNumDenseRf(const NumericMatrix &num) {
  numT = transpose(num);
  blockNum = make_unique<BlockDense<double> >(num.ncol(), numT.begin());
}


unique_ptr<BlockFacRf> BlockFacRf::factory(const List &predBlock) {
  return make_unique<BlockFacRf>(IntegerMatrix((SEXP) predBlock["blockFac"]));
}

unique_ptr<BlockNumRf> BlockNumRf::factory(const List &predBlock) {
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


SEXP BlockSetRf::checkPredblock(const List &predBlock) {
  BEGIN_RCPP
  if (!predBlock.inherits("PredBlock")) {
    stop("Expecting PredBlock");
  }

  if (!Rf_isNull(predBlock["blockFacSparse"])) {
    stop ("Sparse factors:  NYI");
  }
  END_RCPP
}


unique_ptr<BlockSetRf> BlockSetRf::factory(const List& sPredBlock) {
  checkPredblock(sPredBlock);
  return make_unique<BlockSetRf>(
                 BlockNumRf::factory(sPredBlock),
                 BlockFacRf::factory(sPredBlock),
                 as<unsigned int>(sPredBlock["nRow"]));
}


BlockSetRf::BlockSetRf(
               unique_ptr<BlockNumRf> blockNum_,
               unique_ptr<BlockFacRf> blockFac_,
               unsigned int nRow_) :
  blockNum(move(blockNum_)),
  blockFac(move(blockFac_)),
  nRow(nRow_),
  blockSet(make_unique<BlockSet>(blockNum->getNum(),
                                 blockFac->getFac(),
                                 nRow)) {
}
