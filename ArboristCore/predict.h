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
  const vector<size_t> treeOrigin;
  unique_ptr<unsigned int[]> predictLeaves; // Tree-relative leaf indices.


  /**
     @brief Assigns a true leaf index at the prediction coordinates passed.

     @return void.
   */
  inline void predictLeaf(unsigned int blockRow,
                      unsigned int tc,
                      unsigned int leafIdx) {
    predictLeaves[nTree * blockRow + tc] = leafIdx;
  }


  void predictAcross(class LeafFrame* leaf,
                     const class BitMatrix *bag,
                     class Quant *quant = nullptr);

  void predictBlock(unsigned int rowStart,
                    unsigned int rowEnd,
                    const class BitMatrix *bag);

  void predictBlockNum(unsigned int rowStart,
                    unsigned int rowEnd,
                    const class BitMatrix *bag);

  void predictBlockFac(unsigned int rowStart,
                    unsigned int rowEnd,
                    const class BitMatrix *bag);
  
  void predictBlockMixed(unsigned int rowStart,
                    unsigned int rowEnd,
                    const class BitMatrix *bag);
  
  void rowNum(unsigned int row,
              unsigned int blockRow,
              const class TreeNode *treeNode,
              const class BitMatrix *bag);

  void rowFac(unsigned int row,
              unsigned int blockRow,
              const class TreeNode *treeNode,
              const class BVJagged *facSplit,
              const class BitMatrix *bag);
  
  void rowMixed(unsigned int row,
                unsigned int blockRow,
                const class TreeNode *treeNode,
                const class BVJagged *facSplit,
                const class BitMatrix *bag);

 public:  
  static const unsigned int rowBlock = 0x2000;
  
  Predict(const class FramePredict* framePredict_,
          const class Forest* forest_,
          bool validate_);


  static void reg(class LeafFrameReg *leaf,
                  const class Forest *forest,
                  const class BitMatrix *bag,
                  const class FramePredict* framePredict,
                  bool validate,
                  class Quant *quant = nullptr);
  
  static void ctg(class LeafFrameCtg *leaf,
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
