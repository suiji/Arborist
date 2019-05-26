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
   @file blockR.h

   @brief C++ class definitions for managing front end-supplied blocks.

   @author Mark Seligman
 */


#ifndef FRAMEMAPR_BLOCK_R_H
#define FRAMEMAPR_BLOCK_R_H

#include "block.h"

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>

/**
   @brief R-level manager for factor-valued observations.
 */
class BlockFacR {
  const IntegerMatrix facT; // Pins scope of integer transpose.
  unique_ptr<class BlockWindow<unsigned int> > blockFac; // Core-level representation.
 public:

  BlockFacR(const IntegerMatrix &fac);

  /**
     @brief Getter for raw core pointer.
   */
  class BlockWindow<unsigned int>* getFac() {
    return blockFac.get();
  }

  /**
     @brief Instantiates manager from front-end representation.

     @param predFrame summarizes a block of factor-valued predictors.
   */
  static unique_ptr<BlockFacR> factory(const List &predFrame);
};


/**
 @brief Abstract class.
*/
class BlockNumR {
 protected:
  unique_ptr<class BlockWindow<double> > blockNum;
public:

  /**
     @brief Getter for raw pointer to core object.
   */
  class BlockWindow<double>* getNum() {
    return blockNum.get();
  }

  /**
     @brief Instantiates bridge-level representation.

     @param predFrame summarizes a front-end block of numeric observations.
   */
  static unique_ptr<BlockNumR> factory(const List& predFrame);
};


/**
   @brief Compressed representation of numeric data.
 */
class BlockNumDenseR : public BlockNumR {
  NumericMatrix numT; // Pins scope of numerical transpose.

 public:

  BlockNumDenseR(const NumericMatrix &_num);
};


// Dense blocks are transposed by the front end, which is typically
  // a numerical package supporting such operations.  RLE blocks
  // are transposed incrementally, by the core.


/**
   @brief Core object with pinned front-end vectors.
 */
class BlockNumRLER : public BlockNumR {
  const NumericVector &val;
  const IntegerVector &rowStart;
  const IntegerVector &runLength;
  const IntegerVector &predStart;
 public:

  BlockNumRLER(const NumericVector &_val,
                const IntegerVector &_rowStart,
                const IntegerVector &_runLength,
                const IntegerVector &_predStart);

};


#endif
