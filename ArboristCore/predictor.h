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
  @brief Parameters specific to the data set, whether training, testing or predicting.
*/
class Predictor {
  //  static const int defaultInt = -1;
  static double *predProb;
  static int *intBase;
  static int *facCount;
  static int *facSum;
  static int *facCard;
  static int nRow;
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
  static void IntegerBlock(int x[], int _nrow, int _ncol, bool doClone = true);
  static void FactorBlock(int xi[], int _nrow, int _ncol, int levelCount[]);
  static void NumericBlock(double xn[], int _nrow, int _ncol, bool doClone = true);

  /**
     @brief Computes compressed factor index. N.B.:  Implementation relies on factors having highest indices.
     
     @param predIdx is the predictor index.

     @return index of 'predIdx' into factor segment or -1, if not factor-valued.
  */
  static inline int FacIdx(int predIdx) {
    return predIdx >= nPredNum ? predIdx - nPredNum : -1;
  }

  /**
   @brief Computes cardinality of factor-valued predictor.  N.B.:  caller verifies this is a factor.

   @param predIdx is the predictor index.

   @return factor cardinality.
  */
  static inline int FacCard(int predIdx) {
    int facIdx = FacIdx(predIdx);
    return facCard[facIdx];
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
  static inline int NRow() {
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

  static double SplitVal(int _predIdx, int rkLow, int rkHigh);

  static void SetSortAndTies(const int *rank2Row, class PredOrd *predOrd);
  static void OrderByRank(const int *Col, const int *r2r, class PredOrd *dCol, bool ordinals = true);
  static void OrderByRank(const double *xCol, const int *r2r, class PredOrd *dCol);
  static void Factory(const double _predProb[], int _nPred, int _nRow);
  static void DeFactory();
};

#endif

