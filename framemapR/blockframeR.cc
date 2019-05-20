// Copyright (C)  2012-2019  Mark Seligman
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
   @file blockframeR.cc

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman
 */
#include "blockframeR.h"


SEXP BlockFrameR::checkPredframe(const List &predFrame) {
  BEGIN_RCPP
  if (!predFrame.inherits("Frame")) {
    stop("Expecting Frame");
  }

  if (!Rf_isNull(predFrame["blockFacSparse"])) {
    stop ("Sparse factors:  NYI");
  }
  END_RCPP
}


unique_ptr<BlockFrameR> BlockFrameR::factory(const List& sFrame) {
  checkPredframe(sFrame);
  return make_unique<BlockFrameR>(
                 BlockNumR::factory(sFrame),
                 BlockFacR::factory(sFrame),
                 as<unsigned int>(sFrame["nRow"]));
}


BlockFrameR::BlockFrameR(
               unique_ptr<BlockNumR> blockNum_,
               unique_ptr<BlockFacR> blockFac_,
               unsigned int nRow_) :
  blockNum(move(blockNum_)),
  blockFac(move(blockFac_)),
  nRow(nRow_),
  blockFrame(make_unique<BlockFrame>(blockNum->getNum(),
                                     blockFac->getFac(),
                                     nRow)) {
}
