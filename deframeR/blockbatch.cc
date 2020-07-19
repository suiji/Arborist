// Copyright (C)  2012-2020   Mark Seligman
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
   @file blockbatch.cc

   @brief R-style data structures batched as subblocks.

   @author Mark Seligman
*/

#include "blockbatch.h"


template<>
unique_ptr<BlockBatch<IntegerMatrix> > BlockBatch<IntegerMatrix>::unwrap(const List& frame) {
  IntegerMatrix blockFac((SEXP) frame["blockFac"]);

  return make_unique<BlockBatchDense<IntegerMatrix> >(blockFac);
}


template<>
unique_ptr<BlockBatch<NumericMatrix> > BlockBatch<NumericMatrix>::unwrap(const List& frame) {
  List blockNumRLE((SEXP) frame["blockNumRLE"]);

  if (blockNumRLE.length() > 0) {
    NumericVector valNumFE((SEXP) blockNumRLE["valNum"]);
    vector<double> valNum(valNumFE.begin(), valNumFE.end());
    IntegerVector runStartFE((SEXP) blockNumRLE["rowStart"]);
    vector<size_t> runStart(runStartFE.begin(), runStartFE.end());
    IntegerVector runLengthFE((SEXP) blockNumRLE["runLength"]);
    vector<size_t> runLength(runLengthFE.begin(), runLengthFE.end());
    IntegerVector predStartFE((SEXP) blockNumRLE["predStart"]);
    vector<size_t> predStart(predStartFE.begin(), predStartFE.end());
    return make_unique<BlockBatchSparse>(move(valNum),
					 move(runStart),
					 move(runLength),
					 move(predStart));
  }
  else {
    NumericMatrix blockNum((SEXP) frame["blockNum"]);
    return make_unique<BlockBatchDense<NumericMatrix> >(blockNum);
  }
}
