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

  virtual void transpose(unsigned int rowStart, unsigned int rowEnd) = 0;


  inline const unsigned int getNCol() const {
    return nCol;
  }

  /**
     @param rowOff is the offset of a given row.

     @return pointer to base of row contents.
   */  
  inline const double *rowBase(unsigned int rowOff) const {
    return blockNumT + nCol * rowOff;
  }
};


/**
   @brief Encodes block of sparse data.
 */
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
  void transpose(unsigned int rowStart, unsigned int rowEnd);
};


/**
   @brief Crescent analogue of BlockSparse.
 */
class BSCresc {
  const unsigned int nRow;
  const unsigned int nPred;

  vector<unsigned int> predStart; // Starting offset for predictor.
  vector<unsigned int> rowStart; // Starting row of run.
  vector<double> valNum; // Numerical value of run.
  vector<unsigned int> runLength; // Length of run.

  /**
     @brief Pushes a run onto individual component vectors.

     @param val is the numeric value of the run.

     @param rl is the run length.

     @param row is the starting row of the run.
   */
  inline void pushRun(double val,
                      unsigned int rl,
                      unsigned int row) {
    valNum.push_back(val);
    runLength.push_back(rl);
    rowStart.push_back(row);
  }

public:

  BSCresc(unsigned int nRow_,
          unsigned int nPred_);

  /**
     @brief Constructs run vectors from I/P format suppled by front end.

     Reads a sparse representation in which only nonzero values and their
     coordinates are specified.  Constructs internal RLE in which runs of
     arbitrary value are recorded for potential autocompression.

     @param eltsNZ hold the nonzero elements of the sparse representation.

     @param nz are row numbers corresponding to nonzero values.

     @param p has length nCol + 1: index i > 0 gives the raw nonzero offset for
     predictor i-1 and index i == 0 gives the base offset.
   */
  void nzRow(const double eltsNZ[],
             const int nz[],
             const int p[]);

  /**
     @brief Getter for run values.
   */
  const vector<double>& getValNum() {
    return valNum;
  }

  /**
     @brief Getter for starting row offsets;
   */
  const vector<unsigned int>& getRowStart() {
    return rowStart;
  }

  /**
     @brief Getter for run lengths.
   */
  const vector<unsigned int> getRunLength() {
    return runLength;
  }


  /**
     @brief Getter for predictor starting offsets.
   */
  const vector<unsigned int> getPredStart() {
    return predStart;
  }
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
  inline void transpose(unsigned int rowStart, unsigned int rowEnd) {
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
   */
  inline void transpose(unsigned int rowStart, unsigned int rowEnd) {
    blockFacT = feFac + nCol * rowStart;
  }


  /**
     @brief Computes the starting position of a row of transposed
     predictor values.

     @param rowOff is the buffer offset for the row.

     @return pointer to beginning of transposed row.
   */
  inline const unsigned int *rowBase(unsigned int rowOff) const {
    return blockFacT + rowOff * nCol;
  }


  /**
     @brief Getter for column count.

     @return value of nCol.
   */
  inline const unsigned int getNCol() const {
    return nCol;
  }
};

#endif
