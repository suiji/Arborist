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


#ifndef FRAMEMAPR_BLOCKBATCH_H
#define FRAMEMAPR_BLOCKBATCH_H

#include "block.h"

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
using namespace std;

/**
   @brief Blocks containing data read by the front end, parametrized by
   batch type.
 */

template<class batchType>
struct BlockBatch {

  /**
     @brief Wraps column-major R-style matrix as row-major core-style block.
   */
  static unique_ptr<BlockDense<double> > coreBlock(NumericMatrix& blockNum) {
    return make_unique<BlockDense<double> >(blockNum.ncol(), blockNum.nrow(), blockNum.begin());
  }


  /**
     @brief Wraps column-major R-style matrix as row-major core-style block.
   */
  static unique_ptr<BlockDense<unsigned int> > coreBlock(IntegerMatrix& blockFac) {
    return make_unique<BlockDense<unsigned int> >(blockFac.ncol(), blockFac.nrow(), (unsigned int*) blockFac.begin());
  }
  static unique_ptr<BlockBatch<batchType> > unwrap(const List& frame);

  virtual ~BlockBatch() {
  }
  
  virtual batchType transpose(size_t rowStart,
                              size_t extent) = 0;
};


/**
   @brief Dense blocks employ batch containers provided by the front end.
 */
template<class batchType>
struct BlockBatchDense : public BlockBatch<batchType> {

  batchType val; // The value read from front end.

  /**
     @brief Constructor takes ownership of front-end object.
   */
  BlockBatchDense<batchType>(batchType val_) : val(move(val_)) {
  }


  ~BlockBatchDense() {
  }

  
  /**
     @brief Extracts full-column submatrix over specified rows and transposes.

     Copy necessary ut create extent x ncol submatrix.

     @param rowStart is the first row of the submatrix.

     @param extent is the number of rows in the submatrix.

     @return transposed submatrix of dimension ncol x extent.
   */
  batchType transpose(size_t rowStart,
                      size_t extent) {
    if (val.ncol() > 0) {
      batchType window = val(Range(rowStart, rowStart + extent - 1), Range(0, val.ncol() - 1));
      return Rcpp::transpose(window);
    }
    else {
      return batchType(0);
    }
  }
};


/**
   @brief Sparse blocks implement an internal run-length encoding.
 */
template<class batchType>
struct BlockBatchRLE : public BlockBatch<batchType> {
};


/**
  @brief Specialization of RLE to NumericMatrix.

  Employs an internal run-length encoding, as no sparse counterpart
  is available from the front end.  Although the batch container is a
  NumericMatrix, transposition employs a custom implementation.
*/
struct BlockBatchSparse : public BlockBatchRLE<NumericMatrix> {
  unique_ptr<BlockRLE<double> > blockRLE; // Internal encoding.

  BlockBatchSparse(size_t nPred,
                   const double* runVal,
                   const unsigned int* rowStart,
                   const unsigned int* runLength,
                   const unsigned int* predStart) :
    blockRLE(make_unique<BlockRLE<double> >(nPred,
                                            runVal,
                                            rowStart,
                                            runLength,
                                            predStart)) {
  };


  ~BlockBatchSparse() {
  }


  /**
     @brief Transposes a subblock of values copied from this.

     @param rowStart is the starting row of the subblock.

     @param extent is the number of rows in the subblock.

     @return transposed block with dimensions extent x nPred.
   */
  NumericMatrix transpose(size_t rowStart,
                          size_t extent) {
    NumericMatrix window(blockRLE->getNCol(), extent);
    blockRLE->transpose(&window[0], rowStart, extent);

    return window;
  }

};

#endif
