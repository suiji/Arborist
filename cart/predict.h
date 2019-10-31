// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predict.h

   @brief Data structures and methods for prediction.

   @author Mark Seligman
 */

#ifndef CORE_PREDICT_H
#define CORE_PREDICT_H

#include "block.h"
#include "typeparam.h"

#include <vector>
#include <algorithm>


/**
   @brief Data frame specialized for prediction.

   The current implementation supports at most one block of factor-valued
   observations and one block of numeric-valued observations.  Hence the
   class is parametrized by two blocks instead of a more general frame.
 */
class PredictFrame {
  static const size_t rowBlock; // Block size.
  
  class Predict* predict;
  const unsigned int nTree;
  const unsigned int noLeaf;
  const class BlockDense<double>* blockNum;
  const class BlockDense<unsigned int>* blockFac;

  /**
     @brief Aliases a row-prediction method tailored for the frame's
     block structure.
   */
  void (PredictFrame::* predictRow)(size_t, size_t);

  unique_ptr<unsigned int[]> predictLeaves; // Tree-relative leaf indices.

  /**
     @brief Dispatches row prediction in parallel.
   */
  void predictBlock(size_t rowStart);
  
  /**
     @brief Multi-row prediction with predictors of only numeric.

     @param rowStart is the absolute starting row for the block.

     @param rowOff is the block-relative row offset.
  */
  void predictNum(size_t rowStart, size_t rowOff);

  /**
     @brief Multi-row prediction with predictors of only factor type.

     Parameters as above.
  */
  void predictFac(size_t rowStart, size_t rowOff);
  

  /**
     @brief Prediction with predictors of both numeric and factor type.
     Parameters as above.
  */
  void predictMixed(size_t rowStart, size_t rowOff);
  

  /**
     @brief Assigns a true leaf index at the prediction coordinates passed.

     @param blockRow is a block-relative row offset.

     @param tc is the index of the current tree.

     @param leafIdx is the leaf index to record.
   */
  inline void predictLeaf(unsigned int blockRow,
                          unsigned int tc,
                          unsigned int leafIdx) {
    predictLeaves[nTree * blockRow + tc] = leafIdx;
  }


public:
  PredictFrame(class Predict* predict,
               const BlockDense<double>* blockNum_,
               const BlockDense<unsigned int>* blockFac_);

  
  /**
     @brief Specifies size of blocks to be passed by front end.

     @param nRow is the total number of observations.

     @return lesser of internal parameter and number of observations.
   */
  static size_t getBlockRows(size_t nRow);

  
  /**
     @brief Dispatches prediction on a block of rows, by predictor type.

     @param rowStart is the starting row over which to predict.
  */
  void predictAcross(size_t rowStart);


  /**
     @brief Computes number of rows in frame.

     Relies on the property that at least one of the (transposed) blocks has
     non-zero row count and that, if both do, the values agree.

     @return number of rows in frame.
   */
  inline auto getExtent() const {
    return blockNum->getNRow() > 0 ? blockNum->getNRow() : blockFac->getNRow();
  }
  
  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  inline auto getNPredFac() const {
    return blockFac->getNCol(); // Transposed.
  }


  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  inline auto getNPredNum() const {
    return blockNum->getNCol(); // Transposed.
  }


  inline bool isFactor(unsigned int predIdx) const {
    return predIdx >= getNPredNum();
  }
  
  /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
   */
  inline unsigned int getIdx(unsigned int predIdx, bool &predIsFactor) const{
    predIsFactor = isFactor(predIdx);
    return predIsFactor ? predIdx - getNPredNum() : predIdx;
  }

  
  /**
     @brief Indicates whether a given row and tree pair is in-bag.

     @param blockRow is the block-relative row position.

     @param tc is the absolute tree index.

     @param[out] termIdx is the predicted tree-relative index.

     @return whether pair is bagged.
   */
  inline bool isBagged(unsigned int blockRow,
                       unsigned int tc,
                       unsigned int &termIdx) const {
    termIdx = predictLeaves[nTree * blockRow + tc];
    return termIdx == noLeaf;
  }

  /**
     @return base address for (transposed) numeric values at row.
   */
  const double* baseNum(size_t rowOff) const;


  /**
     @return base address for (transposed) factor values at row.
   */
  const unsigned int* baseFac(size_t rowOff) const;
};


/**
   @brief Walks the decision forest for each row in a block, collecting
   predictions.
 */
class Predict {
  const class Bag* bag; // In-bag representation.
  const vector<size_t> treeOrigin; // Jagged accessor of tree origins.
  const struct CartNode* treeNode; // Pointer to base of tree nodes.
  const class BVJagged* facSplit; // Jagged accessor of factor-valued splits.
  class LeafFrame* leaf; // Terminal section of forest.
  class Quant* quant;  // Quantile workplace, as needed.
  const bool oob; // Whether prediction constrained to out-of-bag.

  unique_ptr<PredictFrame> frame;

 public:

  const unsigned int nTree; // # trees used in training.
  const unsigned int noLeaf; // Inattainable leaf index value.
  
  Predict(const class Bag* bag_,
          const class Forest* forest_,
          class LeafFrame* leaf_,
          class Quant* quant_,
          bool oob_);


  /**
     @brief Generic entry from bridge.

     @param frame contains the observations.
   */
  void scoreBlock(const unsigned int predictLeaves[],
                  size_t rowStart,
                  size_t extent) const;

  
  void quantBlock(const PredictFrame* frame,
                  size_t rowStart,
                  size_t extent) const;

  /**
     @brief Prediction of single row with mixed predictor types.

     @param row is the absolute row of data over which a prediction is made.

     @param blockRow is the row's block-relative row index.
  */
  unsigned int rowMixed(unsigned int tIdx,
                        const PredictFrame* frame,
                        const double* rowNT,
                        const unsigned int* rowFT,
                        size_t row);

  
  /**
     @brief Prediction over a single row with factor-valued predictors only.

     Parameters as in mixed case, above.
  */
  unsigned int rowFac(unsigned int tIdx,
              const unsigned int* rowT,
              size_t row);


  /**
     @brief Prediction of a single row with numeric-valued predictors only.

     Parameters as in mixed case, above.
   */
  unsigned int rowNum(unsigned int tIdx,
                      const double* rowT,
                      size_t row);
};

#endif
