// Copyright (C)  2012-2019  Mark Seligman
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
   @file blockRf.h

   @brief C++ class definitions for managing front end-supplied blocks.

   @author Mark Seligman
 */


#ifndef ARBORIST_BLOCK_RF_H
#define ARBORIST_BLOCK_RF_H

#include "block.h"

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>

/**
   @brief Rf-level manager for factor-valued observations.
 */
class BlockFacRf {
  const IntegerMatrix facT; // Pins scope of integer transpose.
  unique_ptr<class BlockDense<unsigned int> > blockFac; // Core-level representation.
 public:

  BlockFacRf(const IntegerMatrix &fac);

  /**
     @brief Getter for raw core pointer.
   */
  class BlockDense<unsigned int>* getFac() {
    return blockFac.get();
  }

  /**
     @brief Instantiates manager from front-end representation.

     @param predBlock summarizes a block of factor-valued predictors.
   */
  static unique_ptr<BlockFacRf> factory(const List &predBlock);
};


/**
 @brief Abstract class.
*/
class BlockNumRf {
 protected:
  unique_ptr<class Block<double> > blockNum;
public:

  /**
     @brief Getter for raw pointer to core object.
   */
  class Block<double>* getNum() {
    return blockNum.get();
  }

  /**
     @brief Instantiates bridge-level representation.

     @param predBlock summarizes a front-end block of numeric observations.
   */
  static unique_ptr<BlockNumRf> factory(const List& predBlock);
};


/**
   @brief Compressed representation of numeric data.
 */
class BlockNumDenseRf : public BlockNumRf {
  NumericMatrix numT; // Pins scope of numerical transpose.

 public:

  BlockNumDenseRf(const NumericMatrix &_num);
};


// Dense blocks are transposed by the front end, which is typically
  // a numerical package supporting such operations.  Sparse blocks
  // are transposed incrementally, by the core.


/**
   @brief Core object with pinned front-end vectors.
 */
class BlockNumSparseRf : public BlockNumRf {
  const NumericVector &val;
  const IntegerVector &rowStart;
  const IntegerVector &runLength;
  const IntegerVector &predStart;
 public:

  BlockNumSparseRf(const NumericVector &_val,
		    const IntegerVector &_rowStart,
		    const IntegerVector &_runLength,
		    const IntegerVector &_predStart);

};


/**
   @brief Captures ownership of BlockSet and component Blocks.
 */
class BlockSetRf {
  const unique_ptr<class BlockNumRf> blockNum;
  const unique_ptr<class BlockFacRf> blockFac;
  const unsigned int nRow;
  const unique_ptr<class BlockSet> blockSet;
 public:

  BlockSetRf(unique_ptr<class BlockNumRf> blockNum_,
             unique_ptr<class BlockFacRf> blockFac_,
             unsigned int nRow);


  /**
     @brief Ensures the passed object has PredBlock type.

     @param predBlock is the object to be checked.
   */
  static SEXP checkPredblock(const List& predBlock);


  /**
     @brief Caches blocks from front end.
   */
  static unique_ptr<BlockSetRf> factory(const List& sPredBlock);

  /**
     @brief Getter for core object pointer.
   */
  const auto getSet() const {
    return blockSet.get();
  }
};


#endif
