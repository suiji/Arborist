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
   @file blockframeR.h

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman

 */

#ifndef FRAMEMAPR_BLOCKFRAME_R_H
#define FRAMEMAPR_BLOCKFRAME_R_H

#include "blockframe.h"
#include "blockR.h"

#include <Rcpp.h>
using namespace Rcpp;


/**
   @brief Captures ownership of BlockFrame and component Blocks.
 */
class BlockFrameR {
  const unique_ptr<class BlockNumR> blockNum;
  const unique_ptr<class BlockFacR> blockFac;
  const unsigned int nRow;
  const unique_ptr<class BlockFrame> blockFrame;
 public:

  BlockFrameR(unique_ptr<class BlockNumR> blockNum_,
             unique_ptr<class BlockFacR> blockFac_,
             unsigned int nRow);


  /**
     @brief Ensures the passed object has PredFrame type.

     @param predFrame is the object to be checked.
   */
  static SEXP checkPredframe(const List& predFrame);


  /**
     @brief Caches blocks from front end.
   */
  static unique_ptr<BlockFrameR> factory(const List& sPredFrame);

  /**
     @brief Getter for core object pointer.
   */
  const auto getFrame() const {
    return blockFrame.get();
  }
};

#endif
