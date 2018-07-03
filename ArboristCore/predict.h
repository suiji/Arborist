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

#ifndef ARBORIST_PREDICT_H
#define ARBORIST_PREDICT_H

#include <vector>
#include <algorithm>

#include "typeparam.h"

class Predict {
  const bool useBag;
  unsigned int noLeaf; // Inattainable leaf index value.
  const class FramePredict *framePredict;
  const class Forest *forest;
  const unsigned int nTree;
  const unsigned int nRow;
  unique_ptr<unsigned int[]> predictLeaves;


  /**
     @brief Assigns a true leaf index at the prediction coordinates passed.

     @return void.
   */
  inline void predictLeaf(unsigned int blockRow,
                      unsigned int tc,
                      unsigned int leafIdx) {
    predictLeaves[nTree * blockRow + tc] = leafIdx;
  }


  void PredictAcross(class Leaf *leaf,
                     const class BitMatrix *bag,
                     class Quant *quant = nullptr);

  void PredictBlock(unsigned int rowStart,
                    unsigned int rowEnd,
                    const class BitMatrix *bag);

  void PredictBlockNum(unsigned int rowStart,
                    unsigned int rowEnd,
                    const class BitMatrix *bag);

  void PredictBlockFac(unsigned int rowStart,
                    unsigned int rowEnd,
                    const class BitMatrix *bag);
  
  void PredictBlockMixed(unsigned int rowStart,
                    unsigned int rowEnd,
                    const class BitMatrix *bag);
  
  void RowNum(unsigned int row,
              unsigned int blockRow,
              const class ForestNode *forestNode,
              const unsigned int *origin,
              const class BitMatrix *bag);

  void RowFac(unsigned int row,
              unsigned int blockRow,
              const class ForestNode *forestNode,
              const unsigned int *origin,
              const class BVJagged *facSplit,
              const class BitMatrix *bag);
  
  void RowMixed(unsigned int row,
                unsigned int blockRow,
                const class ForestNode *forestNode,
                const unsigned int *origin,
                const class BVJagged *facSplit,
                const class BitMatrix *bag);

 public:  
  static const unsigned int rowBlock = 0x2000;
  
  Predict(const class FramePredict* framePredict_,
          const class Forest* forest_,
          bool validate_);


  static void reg(class LeafReg *leaf,
                  const class Forest *forest,
                  const class BitMatrix *bag,
                  const class FramePredict* framePredict,
                  bool validate,
                  class Quant *quant = nullptr);
  
  static void ctg(class LeafCtg *leaf,
                  const class Forest *forest,
                  const class BitMatrix *bag,
                  const class FramePredict* framePredict,
                  bool validate);
  
  
  /**
     @return whether pair is bagged, plus output terminal index.
   */
  inline bool isBagged(unsigned int blockRow,
                       unsigned int tc,
                       unsigned int &termIdx) const {
    termIdx = predictLeaves[nTree * blockRow + tc];
    return termIdx == noLeaf;
  }
};

#endif
