// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictor.h

   @brief Class definitions for maintenance of predictor information.

   @author Mark Seligman
 */

#ifndef ARBORIST_PREDICTOR_H
#define ARBORIST_PREDICTOR_H

/**
   @brief Contains the predictor-specific component of the staged data.
 */
class PredOrd {
 public:
  unsigned int rank; // True rank, with ties identically receiving lowest applicable value.
  unsigned int row; // local copy of r2r[] value.
};


/**
  @brief Parameters specific to the data set, whether training, testing or predicting.
*/
class Predictor {
  //  static const int defaultInt = -1;
  static double *predProb;
  static int *intBase;
  static int *facCount;
  static int *facSum;
  static int *facCard;
  static unsigned int nRow;
  static int nPredInt;
  static int nPredNum;
  static int nPredFac;
  static bool numClone;  // Whether we keep a private copy of numerical data.
  static bool intClone;  //  ..... integer ...
  static bool facClone;  //  ...... factor .....
  static void SetProbabilities(const double _predProb[]);
 public:
  static int *facBase;   // Ordered factor list.
  static double *numBase;
  static int nPred;
  static int nCardTot; // Total number of levels over all factor predictors.
  static int maxFacCard;  // Highest number of levels among all factors.
  static void UniqueRank(int *);
  static void IntegerBlock(int x[], unsigned int _nrow, int _ncol, bool doClone = true);
  static void FactorBlock(int xi[], unsigned int _nrow, int _ncol, int levelCount[]);
  static void NumericBlock(double xn[], unsigned int _nrow, int _ncol, bool doClone = true);
  static int BlockEnd();
  static PredOrd *Order();

  /**
     @brief Computes compressed factor index. N.B.:  Implementation relies on factors having highest indices.
     
     @param predIdx is the predictor index.

     @return index of 'predIdx' into factor segment or -1, if not factor-valued.
  */
  static inline int FacIdx(int predIdx) {
    return predIdx >= PredFacFirst() ? predIdx - PredFacFirst() : -1;
  }

  /**
   @brief Computes cardinality of factor-valued predictor, or zero if not a
   factor.

   @param predIdx is the predictor index.

   @return factor cardinality or zero.
  */
  static inline int FacCard(int predIdx) {
    int facIdx = FacIdx(predIdx);
    return facIdx >= 0 ? facCard[facIdx] : 0;
  }

  /**
     @brief Maximal predictor cardinality.  Useful for allocations.

     @return highest cardinality among predictors.
   */
  static inline int MaxFacCard() {
    return maxFacCard;
  }

  /**
     @brief Computes storage offset of factor-valued predictor.

     @param predIdx is the predictor index.

     @return accumulated count of factors preceding this predictor.
  */
  static inline int FacOffset(int predIdx) {
    int facIdx = FacIdx(predIdx);
    return facSum[facIdx];
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
    return nPred;
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
     @return sum of factor cardinalities.
   */
  static inline int NCardTot() {
    return nCardTot;
  }

  /**
     @brief Fixes contiguous factor ordering as numerical preceding factor.

     @return Position of first numerical predictor.
  */
  static inline int PredNumFirst() {
    return 0;
  }

  /**
     @brief Assumes numerical predictors packed ahead of factor-valued.

     @return Position of last numerical predictor.
  */
  static inline int PredNumSup() {
    return nPredNum;
  }

  /**
     @brief Assumes numerical predictors packed in front of factor-valued.

     @return Position of fist factor-valued predictor.
  */
  static inline int PredFacFirst() {
    return nPredNum;
  }

  /**
     @brief Same assumptions about predictor ordering.

     @return Position of last factor-valued predictor.
  */
  static inline int PredFacSup() {
    return nPred;
  }

  /**
     @brief Determines selection probability for predictor.
     
     @param _predIdx is the predictor index.

     @return probability of selecting predictor.
   */
  static inline double PredProb(int _predIdx) {
    return predProb[_predIdx];
  }


  /**
    @brief Derives split values for a predictor.

    @param predIdx is the preditor index.

    @param rkLow is the lower rank of the split.

    @param rkHigh is the higher rank of the split.

    @return mean predictor value between low and high ranks.
  */
  static inline double SplitVal(int predIdx, int rkLow, int rkHigh) {
    return 0.5 * (numBase[predIdx * nRow + rkLow] + numBase[predIdx * nRow + rkHigh]);
  }

  static void SetSortAndTies(const int *rank2Row, PredOrd *predOrd);
  static void OrderByRank(const int *Col, const int *r2r, PredOrd *dCol, bool ordinals = true);
  static void OrderByRank(const double *xCol, const int *r2r, PredOrd *dCol);
  static void Factory(const double _predProb[], int _nPred, unsigned int _nRow);
  static void DeFactory();
};

#endif

