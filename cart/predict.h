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

#ifndef CART_PREDICT_H
#define CART_PREDICT_H

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
  class Predict* predict;
  const unsigned int nTree;
  const IndexT noLeaf;
  IndexT extent; // # rows in block

  /**
     @brief Gets an acceptable block row count.

     @param rowCount is a requested count.

     @return count of rows in block.
   */
  static size_t getBlockRows(size_t rowCount);


  /**
     @brief Aliases a row-prediction method tailored for the frame's
     block structure.
   */
  void (PredictFrame::* predictRow)(size_t, size_t);

  unique_ptr<IndexT[]> predictLeaves; // Tree-relative leaf indices.

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
                          IndexT leafIdx) {
    predictLeaves[nTree * blockRow + tc] = leafIdx;
  }


public:
  PredictFrame(class Predict* predict,
	       IndexT extent);


  /**
     @brief Dispatches prediction on a block of rows, by predictor type.

     @param rowStart is the starting row over which to predict.
  */
  void predictAcross(size_t rowStart);


  /**
     @brief Indicates whether a given row and tree pair is in-bag.

     @param blockRow is the block-relative row position.

     @param tc is the absolute tree index.

     @param[out] termIdx is the predicted tree-relative index.

     @return whether pair is bagged.
   */
  inline bool isBagged(unsigned int blockRow,
                       unsigned int tc,
                       IndexT& termIdx) const {
    termIdx = predictLeaves[nTree * blockRow + tc];
    return termIdx == noLeaf;
  }
};


/**
   @brief Walks the decision forest for each row in a block, collecting
   predictions.
 */
class Predict {
  static const size_t rowBlock; // Block size.
  
  const class Bag* bag; // In-bag representation.
  const vector<size_t> treeOrigin; // Jagged accessor of tree origins.
  const struct TreeNode* treeNode; // Pointer to base of tree nodes.
  const class BVJagged* facSplit; // Jagged accessor of factor-valued splits.
  class LeafFrame* leaf; // Terminal section of forest.
  struct RLEFrame* rleFrame; // Frame of observations.
  class Quant* quant;  // Quantile workplace, as needed.

  unique_ptr<PredictFrame> frame;


  /**
     @brief Driver for all-row prediction.
   */
  void predictRows();


  /**
     @brief Performs prediction on separately-permuted predictor columns.
   */
  void predictPermute();
  

  /**
     @brief Strip-mines prediction by fixed-size blocks.
   */
  size_t predictBlock(size_t row,
		      size_t extent);


  /**
     @brief Predicts over a single frame of observations.

     @param row is the beginning row index of the block.
   */
  void framePredict(size_t row,
		    size_t extent);


public:

  const PredictorT nPredNum;
  const PredictorT nPredFac;
  const unsigned int nTree; // # trees used in training.
  const IndexT noLeaf; // Inattainable leaf index value.
  vector<unsigned int> trFac; // OTF transposed factor observations.
  vector<double> trNum; // OTF transposed numeric observations.
  vector<size_t> trIdx; // Most recent RLE index accessed by predictor.

  Predict(const class Bag* bag_,
          const class Forest* forest_,
          class LeafFrame* leaf_,
	  struct RLEFrame* rleFrame_,
          class Quant* quant_);

  
  /**
     @brief Main entry from bridge.

     @param importance is true iff permutation importance is specified.

     Distributed prediction will require start and extent parameters.
   */
  void predict(bool importance);

  
    /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
   */
  inline unsigned int getIdx(unsigned int predIdx, bool &predIsFactor) const {
    predIsFactor = isFactor(predIdx);
    return predIsFactor ? predIdx - nPredNum : predIdx;
  }

  
  inline bool isFactor(unsigned int predIdx) const {
    return predIdx >= nPredNum;
  }

  
  /**
     @brief Computes pointer to base of row of numeric values.

     @param rowOff is a block-relative row offset.

     @return base address for numeric values at row.
  */
  const double* baseNum(size_t rowOff) const;


  /**
     @brief As above, but factor varlues.

     @return base address for (transposed) factor values at row.
   */
  const PredictorT* baseFac(size_t rowOff) const;

  
  /**
     @brief Transposes a block of observations to row-major.

     @param rowStart is the starting observation row.
   */
  void transpose(size_t rowStart);


  /**
     @brief Generic entry from bridge.

     @param frame contains the observations.
   */
  void scoreBlock(const IndexT predictLeaves[],
                  size_t rowStart,
                  size_t extent) const;

  
  void quantBlock(const PredictFrame* frame,
                  size_t rowStart,
                  size_t extent) const;

  /**
     @brief Prediction of single row with mixed predictor types.

     @param row is the absolute row of data over which a prediction is made.

     @param blockRow is the row's block-relative row index.

     @return index of leaf predicted.
  */
  IndexT rowMixed(unsigned int tIdx,
		  const double* rowNT,
		  const unsigned int* rowFT,
		  size_t row);

  
  /**
     @brief Prediction over a single row with factor-valued predictors only.

     Parameters as in mixed case, above.
  */
  IndexT rowFac(unsigned int tIdx,
		const unsigned int* rowT,
		size_t row);


  /**
     @brief Prediction of a single row with numeric-valued predictors only.

     Parameters as in mixed case, above.
   */
  IndexT rowNum(unsigned int tIdx,
		const double* rowT,
		size_t row);
};

#endif
