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

#include <vector>
#include <algorithm>

using namespace std;


/**
   @brief Data frame specialized for prediction.

   The current implementation supports at most one block of factor-valued
   observations and one block of numeric-valued observations.  Hence the
   class is parametrized by two blocks instead of a more general frame.
 */
class PredictFrame {
  class Predict* predict;
  const class BlockDense<double>* blockNum;
  const class BlockDense<unsigned int>* blockFac;

  /**
     @brief Aliases a row-prediction method tailored for the frame's
     block structure.
   */
  void (PredictFrame::* predictRow)(size_t, size_t) const;

  /**
     @brief Dispatches row prediction in parallel.
   */
  void predictBlock(size_t rowStart) const;
  
  /**
     @brief Multi-row prediction with predictors of only numeric.

     @param rowStart is the absolute starting row for the block.

     @param rowOff is the block-relative row offset.
  */
  void predictNum(size_t rowStart, size_t rowOff) const;

  /**
     @brief Multi-row prediction with predictors of only factor type.

     Parameters as above.
  */
  void predictFac(size_t rowStart, size_t rowOff) const;
  

  /**
     @brief Prediction with predictors of both numeric and factor type.
     Parameters as above.
  */
  void predictMixed(size_t rowStart, size_t rowOff) const;
  

public:
  PredictFrame(class Predict* predict,
               const BlockDense<double>* blockNum_,
               const BlockDense<unsigned int>* blockFac_);


  /**
     @brief Dispatches prediction on a block of rows, by predictor type.

     @param rowStart is the starting row over which to predict.
  */
  void predictAcross(size_t rowStart) const;


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
  static constexpr size_t rowBlock = 0x2000; // Block size.
  
  const class Bag* bag; // In-bag representation.
  const unsigned int nTree; // # trees used in training.
  const vector<size_t> treeOrigin; // Jagged accessor of tree origins.
  const class TreeNode* treeNode; // Pointer to base of tree nodes.
  const class BVJagged* facSplit; // Jagged accessor of factor-valued splits.
  class LeafFrame* leaf; // Terminal section of forest.
  const unsigned int noLeaf; // Inattainable leaf index value.
  class Quant* quant;  // Quantile workplace, as needed.
  const bool oob; // Whether prediction constrained to out-of-bag.
  unique_ptr<unsigned int[]> predictLeaves; // Tree-relative leaf indices.

  unique_ptr<PredictFrame> frame;

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
  Predict(const class Bag* bag_,
          const class Forest* forest_,
          class LeafFrame* leaf_,
          class Quant* quant_,
          bool oob_);



  /**
     @brief Specifies size of blocks to be passed by front end.

     @param nRow is the total number of observations.

     @return lesser of internal parameter and number of observations.
   */
  static constexpr size_t getBlockRows(size_t nRow) {
    return min(nRow, rowBlock);
  }


  /**
     @brief Generic entry from bridge.

     @param frame contains the observations.
   */
  //  void predict(const PredictFrame* frame, size_t rowStart);

  void scoreBlock(size_t rowStart, size_t extent) const;

  void quantBlock(size_t rowStart, size_t extent) const;

  /**
     @brief Prediction of single row with mixed predictor types.

     @param row is the absolute row of data over which a prediction is made.

     @param blockRow is the row's block-relative row index.

  */
  void rowMixed(const PredictFrame* frame,
                size_t row,
                size_t extent);

  /**
     @brief Prediction over a single row with factor-valued predictors only.

     Parameters as in mixed case, above.
  */
  void rowFac(const PredictFrame* frame,
              size_t row,
              size_t extent);

  /**
     @brief Prediction of a single row with numeric-valued predictors only.

     Parameters as in mixed case, above.
   */
  void rowNum(const PredictFrame* frame,
              size_t row,
              size_t extent);


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
};

#endif
