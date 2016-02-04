// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predblock.h

   @brief Class definitions for maintenance of predictor information.

   @author Mark Seligman
 */

#ifndef ARBORIST_PREDBLOCK_H
#define ARBORIST_PREDBLOCK_H

/**
   @brief For now, all members are static and initialized once per training or
   prediction session.
 */
class PredBlock {
 protected:
  static void DeImmutables();
  static unsigned int nPredNum;
  static unsigned int nPredFac;
  static unsigned int nRow;
 public:  
  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  static inline unsigned int FacFirst() {
    return nPredNum;
  }

  
  /**
     @brief Determines whether predictor is numeric of factor.

     @param predIdx is internal predictor index.

     @return true iff index references a factor.
   */
  static inline bool IsFactor(unsigned int predIdx)  {
    return predIdx >= FacFirst();
  }


  /**
     @brief Computes block-relative position for a predictor.
   */
  static inline unsigned int BlockIdx(int predIdx, bool &isFactor) {
    isFactor = IsFactor(predIdx);
    return isFactor ? predIdx - FacFirst() : predIdx;
  }


  /**
     @return number or observation rows.
   */
  static inline unsigned int NRow() {
    return nRow;
  }

  /**
     @return number of observation predictors.
  */
  static inline int NPred() {
    return nPredFac + nPredNum;
  }

  /**
     @return number of factor predictors.
   */
  static inline int NPredFac() {
    return nPredFac;
  }

  /**
     @return number of numerical predictors.
   */
  static inline int NPredNum() {
    return nPredNum;
  }


  /**
     @brief Fixes contiguous factor ordering as numerical preceding factor.

     @return Position of first numerical predictor.
  */
  static inline int NumFirst() {
    return 0;
  }


  /**
     @brief Positions predictor within numerical block.

     @param predIdx is the core-ordered index of a predictor assumed to be numeric.

     @return Position of predictor within numerical block.
   */
  static inline int NumIdx(int predIdx) {
    return predIdx - NumFirst();
  }


  /**
     @brief Assumes numerical predictors packed ahead of factor-valued.

     @return Position of last numerical predictor.
  */
  static inline int NumSup() {
    return nPredNum;
  }

  
  /**
     @brief Same assumptions about predictor ordering.

     @return Position of last factor-valued predictor.
  */
  static inline int FacSup() {
    return nPredNum + nPredFac;
  }
};


/**
   @brief Training caches numerical predictors for evaluating splits.
 */
class PBTrain : public PredBlock {
  static double *feNum;
  static int *feCard; // Factor predictor cardinalities.
 public:
  static unsigned int cardMax;  // High watermark of factor cardinalities.
  static void Immutables(double *_feNum, int _feCard[], const int _cardMax, const unsigned int _nPredNum, const unsigned int _nPredFac, const unsigned int _nRow);
  static void DeImmutables();
  static double MeanVal(int predIdx, int rowLow, int rowHigh);
  /**
   @brief Computes cardinality of factor-valued predictor, or zero if not a
   factor.

   @param predIdx is the internal predictor index.

   @return factor cardinality or zero.
  */
  static inline int FacCard(int predIdx) {
    return IsFactor(predIdx) ? feCard[predIdx - FacFirst()] : 0;
  }

  
  /**
     @brief Maximal predictor cardinality.  Useful for packing.

     @return highest cardinality, if any, among factor predictors.
   */
  static inline int CardMax() {
    return cardMax;
  }
};


class PBPredict : public PredBlock {
 public:
  static double *feNumT;
  static int *feFacT;

  static void Immutables(double *_feNumT, int *_feFacT, const unsigned int _nPredNum, const unsigned int _nPredFac, const unsigned int _nRow);

  static void DeImmutables();

  /**
     @return base address for (transposed) numeric values at row.
   */
  static inline double *RowNum(unsigned int row) {
    return &feNumT[nPredNum * row];
  }


  /**
     @return base address for (transposed) factor values at row.
   */
  static inline int *RowFac(unsigned int row) {
    return &feFacT[nPredFac * row];
  }

};

#endif
