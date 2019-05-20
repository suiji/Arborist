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
  const unsigned int nCol; // # columns in untransposed form.
 public:

  Block(const ty raw_[],
        unsigned int nCol_) :
    raw(raw_),
    nCol(nCol_) {}

  virtual ~Block() {}

  inline const auto getNCol() const {
    return nCol;
  }

};


template<class ty>
class BlockDense : public Block<ty> {
public:

  BlockDense(unsigned int nCol,
             const ty raw_[]) :
    Block<ty>(raw_, nCol) {
  }

  ~BlockDense() {
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
              unsigned int nCol_) :
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
   @brief Variant offering sub-block windowing.  Temporary workaround
   to be removed when blocked transposition is available.
 */
template<class ty>
class BlockWindow : public Block<ty> {
protected:
  unsigned int rowWindow; // Iterator state.

public:
  
  /**
     @brief Updates window offset.
   */
  virtual inline void reWindow(unsigned int rowStart,
                               unsigned int rowEnd,
                               unsigned int rowBlock) {
    rowWindow = rowStart;
  }


  /**
     @brief Determines position of row within window.

     @param rowOff is the offset of a given row.

     @return pointer to base of row contents.
   */  
  virtual inline const ty* rowBase(unsigned int rowOff) const {
    return Block<ty>::raw + Block<ty>::nCol * (rowWindow + rowOff);
  }

  BlockWindow(unsigned int nCol,
              const ty raw_[]) :
    Block<ty>(raw_, nCol) {
  }


  ~BlockWindow() {
  }
};
  

/**
   @brief Runlength-encoded sparse representation.
 */
template<class ty>
class BlockWindowRLE : public BlockWindow<ty> {
  const unsigned int* rowStart;
  const unsigned int* runLength;
  const unsigned int* predStart;
  vector<unsigned int> rowNext;
  vector<unsigned int> idxNext;
  ty* window;  // iterator state.
  vector<ty> transVal; // iterator work space.

public:

 /**
     @brief Sparse constructor for prediction frame.
  */
  BlockWindowRLE(unsigned int nCol_,
           const ty* raw_,
           const unsigned int* rowStart_,
           const unsigned int* runLength_,
           const unsigned int* predStart_) :
    BlockWindow<ty>(nCol_, raw_),
    rowStart(rowStart_),
    runLength(runLength_),
    predStart(predStart_),
    rowNext(vector<unsigned int>(Block<ty>::nCol)),
    idxNext(vector<unsigned int>(Block<ty>::nCol)),
    window(nullptr),
    transVal(vector<ty>(Block<ty>::nCol)) {
    fill(rowNext.begin(), rowNext.end(), 0ul); // Position of first update.
    unsigned int predIdx = 0;
    for (auto & idxN : idxNext) {
      idxN = predStart[predIdx++]; // Current starting offset.
    }
  }

  ~BlockWindowRLE() {
    if (window != nullptr) {
      delete [] window;
    }
  }

  inline void reWindow(unsigned int rowWindow,
                       unsigned int rowEnd,
                       unsigned int rowBlock) {
    BlockWindow<ty>::rowWindow = rowWindow;
    if (window == nullptr) {
      window = new ty[rowBlock * Block<ty>::nCol];
    }
    for (unsigned int row = rowWindow; row < rowEnd; row++) {
      for (unsigned int predIdx = 0; predIdx < Block<ty>::nCol; predIdx++) {
        if (row == rowNext[predIdx]) { // Assignments persist across invocations:
          unsigned int vecIdx = idxNext[predIdx];
          transVal[predIdx] = Block<ty>::raw[vecIdx];
          rowNext[predIdx] = rowStart[vecIdx] + runLength[vecIdx];
          idxNext[predIdx] = ++vecIdx;
        }
        window[(row - rowWindow) * Block<ty>::nCol + predIdx] = transVal[predIdx];
      }
    }
  }

  /**
     @brief Determines position of row within window.

     @param rowOff is the window-relative offset of a given row.

     @return pointer to base of row contents.
   */  
  inline const ty* rowBase(unsigned int rowOff) const {
    return window + Block<ty>::nCol * rowOff;
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

  BlockIPCresc(unsigned int nRow_,
                unsigned int nCol) :
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
