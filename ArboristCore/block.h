// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file block.h

   @brief Class definitions for maintenance of type-based data blocks.

   @author Mark Seligman
 */

#ifndef ARBORIST_BLOCK_H
#define ARBORIST_BLOCK_H

#include <vector>
#include <cmath>

#include "typeparam.h"


/**
   @brief Abstract class for blocks of predictor values.
 */
class BlockNum {
 protected:
  double *blockNumT; // Iterator state
  const unsigned int nCol; // # columns in untransposed form.
 public:

 BlockNum(unsigned int _nCol) : nCol(_nCol) {}
  virtual ~BlockNum() {}

  static BlockNum *Factory(const vector<double> &_valNum, const vector<unsigned int> &_rowStart, const vector<unsigned int> &_runLength, const vector<unsigned int> &_predStart, double *_feNumT, unsigned int _nCol);

  virtual void Transpose(unsigned int rowStart, unsigned int rowEnd) = 0;


  inline const unsigned int NCol() const {
    return nCol;
  }

  
  inline const double *Row(unsigned int rowOff) const {
    return blockNumT + nCol * rowOff;
  }
};


class BlockSparse : public BlockNum {
  const double *val;
  const unsigned int *rowStart;
  const unsigned int *runLength;
  const unsigned int *predStart;
  double *transVal;
  unsigned int *rowNext;
  unsigned int *idxNext;

 public:

  /**
     @brief Sparse constructor.
   */
  BlockSparse(const double *_val,
	      const unsigned int *_rowStart,
	      const unsigned int *__runLength,
	      const unsigned int *_predStart,
	      unsigned int _nCol);
  ~BlockSparse();
  void Transpose(unsigned int rowStart, unsigned int rowEnd);
};


class BlockNumDense : public BlockNum {
  double *feNumT;
 public:


 BlockNumDense(double *_feNumT,
	       unsigned int _nCol) :
  BlockNum(_nCol) {
    feNumT = _feNumT;
    blockNumT = _feNumT;
  }


  ~BlockNumDense() {
  }

  
  /**
     @brief Resets starting position to block within region previously
     transposed.

     @param rowStart is the first row of the block.

     @param rowEnd is the sup row.  Unused here.

     @return void.
   */
  inline void Transpose(unsigned int rowStart, unsigned int rowEnd) {
    blockNumT = feNumT + nCol * rowStart;
  }
};


class BlockFac {
  const unsigned int nCol;
  unsigned int *feFac; // Factors, may or may not already be transposed.
  unsigned int *blockFacT; // Iterator state.

 public:

  /**
     @brief Dense constructor:  currently pre-transposed.
   */
 BlockFac(unsigned int *_feFacT,
	  unsigned int _nCol) :
  nCol(_nCol),
    feFac(_feFacT) {
    }

  static BlockFac *Factory(unsigned int *_feFacT, unsigned int _nCol);
  
  /**
     @brief Resets starting position to block within region previously
     transposed.

     @param rowStart is the first row of the block.

     @param rowEnd is the sup row.  Unused here.

     @return void.
   */
  inline void Transpose(unsigned int rowStart, unsigned int rowEnd) {
    blockFacT = feFac + nCol * rowStart;
  }


  /**
     @brief Computes the starting position of a row of transposed
     predictor values.

     @param rowOff is the buffer offset for the row.

     @return pointer to beginning of transposed row.
   */
  inline const unsigned int *Row(unsigned int rowOff) const {
    return blockFacT + rowOff * nCol;
  }


  inline const unsigned int NCol() const {
    return nCol;
  }
};

#endif
