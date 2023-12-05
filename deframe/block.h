// This file is part of deframe

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file block.h

   @brief Class definitions for maintenance of type-based data blocks.

   @author Mark Seligman
 */

#ifndef DEFRAME_BLOCK_H
#define DEFRAME_BLOCK_H

#include <vector>
#include <cmath>

using namespace std;


/**
   @brief Abstract class for blocks of predictor values.
 */

template<class ty>
class Block {
 protected:
  const vector<ty> raw;
  const size_t nCol; // # columns.
 public:

  Block(const vector<ty> raw_,
        size_t nCol_) :
    raw(raw_),
    nCol(nCol_) {}

  virtual ~Block() = default;

  const auto getNCol() const {
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
             const vector<ty> raw_) :
    Block<ty>(raw_, nCol_), nRow(nRow_) {
  }

  ~BlockDense() = default;


  size_t getNRow() const {
    return nRow;
  }

  /**
     @brief Exposes contents of a given row.

     @param row is the given row.

     @return pointer to base of row contents.
   */  
  const ty* rowBase(size_t row) const {
    return &Block<ty>::raw[Block<ty>::nCol * row];
  }
};


/**
   @brief Sparse predictor-ranked numerical block.
 */
template<class ty>
class BlockJagged : public Block<ty> {
  const vector<size_t> height; // Accumulated length of each column.

 public:
  BlockJagged(const vector<ty>& raw_,
	      const vector<size_t>& height_) :
    Block<ty>(raw_, raw_.size()),
    height(height_) {
  }


  /**
     @brief Instantiates contents as vector-of-vectors rather than BlockJagged object.
   */
  static vector<vector<ty>> unwrap(const vector<ty>& val,
				   const vector<size_t>& height) {
    vector<vector<ty>> vv(height.size());
    size_t col = 0;
    size_t i = 0;
    for (auto count : height) {
      for (; i < count; i++) {
	vv[col].push_back(val[i]);
      }
      col++;
    }
    return vv;
  }

  /**
     @return rank of specified predictor at specified rank.
   */
  auto getVal(unsigned int predIdx,
                     size_t rk) const {
    return Block<ty>::raw[rk + (predIdx == 0 ? 0 : height[predIdx-1])];
  }
};



/**
   @brief Runlength-encoded sparse representation.
 */
template<class ty>
class BlockRLE : public Block<ty> {
  const vector<size_t> runStart;
  const vector<size_t> runLength;
  const vector<size_t> predStart;
  // Persistent transpose state:
  vector<size_t> rowNext;
  vector<size_t> idxNext;
  vector<ty> transVal;

public:

 /**
     @brief Sparse constructor for prediction frame.
  */
  BlockRLE(const vector<ty>& raw_,
           const vector<size_t>& runStart_,
           const vector<size_t>& runLength_,
           const vector<size_t>& predStart_) :
    Block<ty>(raw_, predStart_.size()),
    runStart(runStart_),
    runLength(runLength_),
    predStart(predStart_),
    rowNext(vector<size_t>(Block<ty>::nCol)),
    idxNext(vector<size_t>(Block<ty>::nCol)),
    transVal(vector<ty>(Block<ty>::nCol)) {
    fill(rowNext.begin(), rowNext.end(), 0ul); // Position of first update.
    unsigned int predIdx = 0;
    for (auto & idxN : idxNext) {
      idxN = predStart[predIdx++]; // Current starting offset.
    }
  }

  ~BlockRLE() = default;

  
  /**
     @brief Transposes a block of rows into a dense sub-block.

     @param[out] window outputs the densely-transposed values.
   */  
  void transpose(ty* window,
                        size_t rowStart,
                        size_t extent) {
    ty* winRow = window;
    for (size_t row = rowStart; row != rowStart + extent; row++) {
      for (unsigned int predIdx = 0; predIdx < Block<ty>::nCol; predIdx++) {
        if (row == rowNext[predIdx]) { // Assignments persist across invocations:
          size_t valIdx = idxNext[predIdx];
          transVal[predIdx] = Block<ty>::raw[valIdx];
          rowNext[predIdx] = runStart[valIdx] + runLength[valIdx];
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
  const size_t nRow;
  const unsigned int nPred;

  vector<size_t> predStart; // Starting element per predictor.
  vector<size_t> runStart; // Starting row of run.
  vector<ty> val; // Value of run.
  vector<size_t> runLength; // Length of run.

  /**
     @brief Pushes a run onto individual component vectors.

     @param runVal is the value of the run.

     @param rl is the run length.

     @param row is the starting row of the run.
   */
  void pushRun(ty runVal,
                      size_t rl,
                      size_t row) {
    val.push_back(runVal);
    runLength.push_back(rl);
    runStart.push_back(row);
  }

public:

  BlockIPCresc(size_t nRow_,
               size_t nCol) :
    nRow(nRow_),
    nPred(nCol),
    predStart(vector<size_t>(nPred)) {
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
  const vector<size_t>& getRunStart() const {
    return runStart;
  }

  /**
     @brief Getter for run lengths.
   */
  const vector<size_t> getRunLength() const {
    return runLength;
  }


  /**
     @brief Getter for predictor starting offsets.
   */
  const vector<size_t> getPredStart() const {
    return predStart;
  }

  /**
     @brief Constructs run vectors from I/P format suppled by front end.

     Reads a sparse representation in which only nonzero values and their
     coordinates are specified.  Constructs internal RLE in which runs of
     arbitrary value are recorded for potential autocompression.

     @param eltsNZ hold the nonzero elements of the sparse representation.

     @param rowNZ are row numbers corresponding to nonzero values.

     @param idxPred has length nCol + 1: index i > 0 gives the raw nonzero offset for
     predictor i-1 and index i == 0 gives the base offset.
   */
  void nzRow(const ty eltsNZ[],
             const vector<size_t>& rowNZ,
             const vector<size_t>& idxPred) {
    const ty zero = 0.0;
    for (unsigned int colIdx = 0; colIdx < nPred; colIdx++) {
      predStart[colIdx] = val.size();
      auto nzHeight = idxPred[colIdx + 1] - idxPred[colIdx];
      if (nzHeight == 0) { // No nonzero values for predictor.
        pushRun(zero, nRow, 0);
      }
      else {
        auto nzPrev = nRow; // Inattainable row value.
        for (size_t rowIdx = idxPred[colIdx]; rowIdx != idxPred[colIdx] + nzHeight; rowIdx++) {
          auto nzRow = rowNZ[rowIdx]; // row # of nonzero element.
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
