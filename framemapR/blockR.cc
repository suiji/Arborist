// Copyright (C)  2012-2019   Mark Seligman
//
// This file is part of framemapR.
//
// framemapR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// framemapR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with framemapR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file blockR.cc

   @brief C++ interface to R entries for maintaining predictor data structures.

   @author Mark Seligman
*/

#include "blockR.h"


BlockFacR::BlockFacR(const IntegerMatrix &fac) :
  facT(transpose(fac)) {
  blockFac = make_unique<BlockWindow<unsigned int> >(fac.ncol(), (unsigned int*)facT.begin());
}


/**
  @brief RLE constructor.
 */
BlockNumRLER::BlockNumRLER(const NumericVector &_val,
                             const IntegerVector &_rowStart,
                             const IntegerVector &_runLength,
                             const IntegerVector &_predStart) :
  val(_val),
  rowStart(_rowStart),
  runLength(_runLength),
  predStart(_predStart) {
  blockNum = make_unique<BlockWindowRLE<double> >(predStart.length(),
                                            val.begin(),
                              (const unsigned int *) rowStart.begin(),
			      (const unsigned int *) runLength.begin(),
                              (const unsigned int *) predStart.begin());
}


/**
   @brief Dense constructor
 */
BlockNumDenseR::BlockNumDenseR(const NumericMatrix &num) {
  numT = transpose(num);
  blockNum = make_unique<BlockWindow<double> >(num.ncol(), numT.begin());
}


unique_ptr<BlockFacR> BlockFacR::factory(const List &predFrame) {
  return make_unique<BlockFacR>(IntegerMatrix((SEXP) predFrame["blockFac"]));
}

unique_ptr<BlockNumR> BlockNumR::factory(const List &predFrame) {
  List blockNumRLE((SEXP) predFrame["blockNumSparse"]);
  if (blockNumRLE.length() > 0) {
    return make_unique<BlockNumRLER>(
		  NumericVector((SEXP) blockNumRLE["valNum"]),
		  IntegerVector((SEXP) blockNumRLE["rowStart"]),
		  IntegerVector((SEXP) blockNumRLE["runLength"]),
		  IntegerVector((SEXP) blockNumRLE["predStart"]));
  }
  else {
    return make_unique<BlockNumDenseR>(
				 NumericMatrix((SEXP) predFrame["blockNum"])
					 );
  }
}
