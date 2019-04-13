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
   @file blockBridge.h

   @brief C++ class definitions for managing front end-supplied blocks.

   @author Mark Seligman
 */


#ifndef ARBORIST_BLOCK_BRIDGE_H
#define ARBORIST_BLOCK_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
#include "block.h"

/**
   @brief Bridge-level manager for factor-valued observations.
 */
class BlockFacBridge {
  const IntegerMatrix facT; // Pins scope of integer transpose.
  unique_ptr<BlockFac> blockFac; // Core-level representation.
 public:

  BlockFacBridge(const IntegerMatrix &fac);

  /**
     @brief Getter for raw core pointer.
   */
  BlockFac *getFac() {
    return blockFac.get();
  }

  /**
     @brief Instantiates manager from front-end representation.

     @param predBlock summarizes a block of factor-valued predictors.
   */
  static unique_ptr<BlockFacBridge> Factory(const List &predBlock);
};


/**
 @brief Abstract class.
*/
class BlockNumBridge {
 protected:
  unique_ptr<BlockNum> blockNum;
public:

  /**
     @brief Getter for raw pointer to core object.
   */
  BlockNum *getNum() {
    return blockNum.get();
  }

  /**
     @brief Instantiates bridge-level representation.

     @param predBlock summarizes a front-end block of numeric observations.
   */
  static unique_ptr<BlockNumBridge> Factory(const List& predBlock);
};


/**
   @brief Compressed representation of numeric data.
 */
class BlockDenseBridge : public BlockNumBridge {
  NumericMatrix numT; // Pins scope of numerical transpose.

 public:

  BlockDenseBridge(const NumericMatrix &_num);
};


// Dense blocks are transposed by the front end, which is typically
  // a numerical package supporting such operations.  Sparse blocks
  // are transposed incrementally, by the core.


/**
   @brief Core object with pinned front-end vectors.
 */
class BlockSparseBridge : public BlockNumBridge {
  const NumericVector &val;
  const IntegerVector &rowStart;
  const IntegerVector &runLength;
  const IntegerVector &predStart;
 public:

  BlockSparseBridge(const NumericVector &_val,
		    const IntegerVector &_rowStart,
		    const IntegerVector &_runLength,
		    const IntegerVector &_predStart);

};

#endif
