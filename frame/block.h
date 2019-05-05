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

using namespace std;


/**
   @brief Abstract class for blocks of predictor values.
 */

template<class ty>
class Block {
 protected:
  ty* blockT; // Iterator state
  const unsigned int nCol; // # columns in untransposed form.
 public:

  Block(ty* blockT_,
        unsigned int nCol_) : blockT(blockT_),
                              nCol(nCol_) {}
  virtual ~Block() {}

  virtual void transpose(unsigned int rowStart,
                         unsigned int rowEnd,
                         unsigned int rowBlock) = 0;

  inline const auto getNCol() const {
    return nCol;
  }

  /**
     @param rowOff is the offset of a given row.

     @return pointer to base of row contents.
   */  
  inline const auto rowBase(unsigned int rowOff) const {
    return blockT + nCol * rowOff;
  }
};

template<class ty>
class BlockDense : public Block<ty> {
  ty* feT;

public:

  BlockDense(unsigned int nCol_,
             ty* feT_) :
    Block<ty>(feT_, nCol_),
    feT(feT_) {
  }

  ~BlockDense() {
  }

  
  /**
     @brief Resets starting position to block within region previously
     transposed.

     @param rowStart is the first row of the block.

     @param rowEnd is the sup row.  Unused here.
   */
  inline void transpose(unsigned int rowStart,
                        unsigned int rowEnd,
                        unsigned int rowBlock) {
    Block<ty>::blockT = feT + Block<ty>::nCol * rowStart;
  }
};

/**
   @brief Encodes block of sparse data.
 */
template<class ty>
class BlockSparse : public Block<ty> {
  const ty* val;
  const unsigned int* rowStart;
  const unsigned int* runLength;
  const unsigned int* predStart;
  ty* transVal;
  unsigned int* rowNext;
  unsigned int* idxNext;

public:

 /**
     @brief Sparse constructor for prediction frame.
  */
  BlockSparse(unsigned int nCol_,
              const ty* val_,
              const unsigned int* rowStart_,
              const unsigned int* runLength_,
              const unsigned int* predStart_) :
    Block<ty>(nullptr, nCol_),
    val(val_),
    rowStart(rowStart_),
    runLength(runLength_),
    predStart(predStart_),
    transVal(nullptr) {

  // Both 'blockNumT' and 'valPrev' are updated before the next use, so
  // need not be initialized.
    rowNext = new unsigned int[Block<ty>::nCol];
    idxNext = new unsigned int[Block<ty>::nCol];
    for (unsigned int predIdx = 0; predIdx < Block<ty>::nCol; predIdx++) {
      rowNext[predIdx] = 0; // Position of first update.
      idxNext[predIdx] = predStart[predIdx]; // Current starting offset.
    }
  }

  ~BlockSparse() {
    if (transVal != nullptr) {
      delete [] transVal;
      delete [] Block<ty>::blockT;
    }
    delete [] rowNext;
    delete [] idxNext;
  }

  void transpose(unsigned int rowBegin,
                 unsigned int rowEnd,
                 unsigned int rowBlock) {
    if (Block<ty>::blockT == nullptr) {
      Block<ty>::blockT = new ty[rowBlock * Block<ty>::nCol];
      transVal = new ty[Block<ty>::nCol];
    }
    for (unsigned int row = rowBegin; row < rowEnd; row++) {
      for (unsigned int predIdx = 0; predIdx < Block<ty>::nCol; predIdx++) {
        if (row == rowNext[predIdx]) { // Assignments persist across invocations:
          unsigned int vecIdx = idxNext[predIdx];
          transVal[predIdx] = val[vecIdx];
          rowNext[predIdx] = rowStart[vecIdx] + runLength[vecIdx];
          idxNext[predIdx] = ++vecIdx;
        }
        Block<ty>::blockT[(row - rowBegin) * Block<ty>::nCol + predIdx] = transVal[predIdx];
      }
    }
  }
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
  const vector<double>& getValNum() const {
    return valNum;
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
};


/**
   @brief Collection of variously typed blocks of contiguous storage.

   Currently implemented as numeric and factor only, but may potentially
   support arbitrary collections.
 */

class BlockSet {
  Block<double>* blockNum;
  BlockDense<unsigned int>* blockFac;
  unsigned int nRow;

public:
  BlockSet(Block<double>* blockNum_,
           BlockDense<unsigned int>* blockFac,
           unsigned nRow_);

  /**
     @brief Accessor for row count.
   */
  inline auto getNRow() const {
    return nRow;
  }
  
  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  inline unsigned int getNPredFac() const {
    return blockFac->getNCol();
  }


  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  inline unsigned int getNPredNum() const {
    return blockNum->getNCol();
  }

  
  /**
     @brief Determines whether predictor is numeric or factor.

     @param predIdx is internal predictor index.

     @return true iff index references a factor.
   */
  inline bool isFactor(unsigned int predIdx)  const {
    return predIdx >= getNPredNum();
  }


  /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
   */
  inline unsigned int getIdx(unsigned int predIdx, bool &thisIsFactor) const{
    thisIsFactor = isFactor(predIdx);
    return thisIsFactor ? predIdx - getNPredNum() : predIdx;
  }


  void transpose(unsigned int rowStart,
                 unsigned int rowEnd,
                 unsigned int rowBlock) const;

  /**
     @return base address for (transposed) numeric values at row.
   */
  const double* baseNum(unsigned int rowOff) const;


  /**
     @return base address for (transposed) factor values at row.
   */
  const unsigned int* baseFac(unsigned int rowOff) const;
};

#endif
