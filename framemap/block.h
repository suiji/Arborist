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

#ifndef FRAMEMAP_BLOCK_H
#define FRAMEMAP_BLOCK_H

#include <vector>
#include <cmath>

using namespace std;


/**
   @brief Abstract class for blocks of predictor values.
 */

template<class ty>
class Block {
 protected:
  const ty* raw;
  const size_t nCol; // # columns.
 public:

  Block(const ty raw_[],
        size_t nCol_) :
    raw(raw_),
    nCol(nCol_) {}

  virtual ~Block() {}

  inline const auto getNCol() const {
    return nCol;
  }
};


/**
   @brief Rectangular block, parametrized by row and column.

   Row-major access.
 */
template<class ty>
class BlockDense : public Block<ty> {
  size_t nRow;

public:

  BlockDense(size_t nRow_,
             size_t nCol_,
             const ty raw_[]) :
    Block<ty>(raw_, nCol_), nRow(nRow_) {
  }

  ~BlockDense() {
  }

  size_t getNRow() const {
    return nRow;
  }

  /**
     @brief Exposes contents of a given row.

     @param row is the given row.

     @return pointer to base of row contents.
   */  
  inline const ty* rowBase(size_t row) const {
    return Block<ty>::raw + Block<ty>::nCol * row;
  }
};


/**
   @brief Sparse predictor-ranked numerical block.
 */
template<class ty>
class BlockJagged : public Block<ty> {
  const unsigned int* colOffset;

 public:
  BlockJagged(const ty raw_[],
	      const unsigned int colOffset_[],
              size_t nCol_) :
    Block<ty>(raw_, nCol_),
    colOffset(colOffset_) {
  }


  /**
     @return rank of specified predictor at specified rank.
   */
  inline auto getVal(unsigned int predIdx,
                     unsigned int rk) const {
    return Block<ty>::raw[colOffset[predIdx] + rk];
  }
};



/**
   @brief Runlength-encoded sparse representation.
 */
template<class ty>
class BlockRLE : public Block<ty> {
  const unsigned int* rowOff;
  const unsigned int* runLength;
  const unsigned int* predStart;
  // Persistent transpose state:
  vector<unsigned int> rowNext;
  vector<unsigned int> idxNext;
  vector<ty> transVal;

public:

 /**
     @brief Sparse constructor for prediction frame.
  */
  BlockRLE(size_t nCol_,
           const ty* raw_,
           const unsigned int* rowOff_,
           const unsigned int* runLength_,
           const unsigned int* predStart_) :
    Block<ty>(raw_, nCol_),
    rowOff(rowOff_),
    runLength(runLength_),
    predStart(predStart_),
    rowNext(vector<unsigned int>(Block<ty>::nCol)),
    idxNext(vector<unsigned int>(Block<ty>::nCol)),
    transVal(vector<ty>(Block<ty>::nCol)) {
    fill(rowNext.begin(), rowNext.end(), 0ul); // Position of first update.
    unsigned int predIdx = 0;
    for (auto & idxN : idxNext) {
      idxN = predStart[predIdx++]; // Current starting offset.
    }
  }

  ~BlockRLE() {
  }


  /**
     @brief Transposes a block of rows into a dense sub-block.

     @param[out] window outputs the densely-transposed values.
   */  
  inline void transpose(ty* window,
                        size_t rowStart,
                        size_t extent) {
    ty* winRow = window;
    for (size_t row = rowStart; row < rowStart + extent; row++) {
      for (unsigned int predIdx = 0; predIdx < Block<ty>::nCol; predIdx++) {
        if (row == rowNext[predIdx]) { // Assignments persist across invocations:
          unsigned int valIdx = idxNext[predIdx];
          transVal[predIdx] = Block<ty>::raw[valIdx];
          rowNext[predIdx] = rowOff[valIdx] + runLength[valIdx];
          idxNext[predIdx] = valIdx + 1;
        }
        winRow[predIdx] = transVal[predIdx];
      }
      winRow += Block<ty>::nCol;
    }
  }
};


/**
   @brief Crescent form of column-compressed sparse block.
 */
template<class ty>
class BlockIPCresc {
  const unsigned int nRow;
  const unsigned int nPred;

  vector<unsigned int> predStart; // Starting offset for predictor.
  vector<unsigned int> rowStart; // Starting row of run.
  vector<ty> val; // Value of run.
  vector<unsigned int> runLength; // Length of run.

  /**
     @brief Pushes a run onto individual component vectors.

     @param runVal is the value of the run.

     @param rl is the run length.

     @param row is the starting row of the run.
   */
  inline void pushRun(ty runVal,
                      unsigned int rl,
                      unsigned int row) {
    val.push_back(runVal);
    runLength.push_back(rl);
    rowStart.push_back(row);
  }

public:

  BlockIPCresc(size_t nRow_,
               size_t nCol) :
    nRow(nRow_),
    nPred(nCol),
    predStart(vector<unsigned int>(nPred)) {
  }


  /**
     @brief Getter for run values.
   */
  const vector<ty>& getVal() const {
    return val;
  }

  /**
     @brief Getter for starting row offsets;
   */
  const vector<unsigned int>& getRowStart() const {
    return rowStart;
  }

  /**
     @brief Getter for run lengths.
   */
  const vector<unsigned int> getRunLength() const {
    return runLength;
  }


  /**
     @brief Getter for predictor starting offsets.
   */
  const vector<unsigned int> getPredStart() const {
    return predStart;
  }

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
  void nzRow(const ty eltsNZ[],
             const int nz[],
             const int p[]) {
  // Pre-scans column heights.
    const ty zero = 0.0;
    vector<unsigned int> nzHeight(nPred + 1);
    unsigned int idxStart = p[0];
    for (unsigned int colIdx = 1; colIdx <= nPred; colIdx++) {
      nzHeight[colIdx - 1] = p[colIdx] - idxStart;
      idxStart = p[colIdx];
    }

    for (unsigned int colIdx = 0; colIdx < predStart.size(); colIdx++) {
      unsigned int colHeight = nzHeight[colIdx]; // # nonzero values in column.
      predStart[colIdx] = val.size();
      if (colHeight == 0) { // No nonzero values for predictor.
        pushRun(zero, nRow, 0);
      }
      else {
        unsigned int nzPrev = nRow; // Inattainable row value.
        // Row indices into 'i' and 'x' are zero-based.
        unsigned int idxStart = p[colIdx];
        unsigned int idxEnd = idxStart + colHeight;
        for (unsigned int rowIdx = idxStart; rowIdx < idxEnd; rowIdx++) {
          unsigned int nzRow = nz[rowIdx]; // row # of nonzero element.
          if (nzPrev == nRow && nzRow > 0) { // Zeroes lead.
            pushRun(zero, nzRow, 0);
          }
          else if (nzRow > nzPrev + 1) { // Zeroes precede.
            pushRun(zero, nzRow - (nzPrev + 1), nzPrev + 1);
          }
          pushRun(eltsNZ[rowIdx], 1, nzRow);
          nzPrev = nzRow;
        }
        if (nzPrev + 1 < nRow) { // Zeroes trail.
          pushRun(zero, nRow - (nzPrev + 1), nzPrev + 1);
        }
      }
    }
  }
};


#endif
