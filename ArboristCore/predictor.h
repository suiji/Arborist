// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_PREDICTOR_H
#define ARBORIST_PREDICTOR_H

// Parameters specific to the data set, whether training, testing or predicting.
//
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

  // Returns index of 'predIdx' into factor segment.
  // N.B.:  Implementation relies on factors having highest indices.
  //
  static inline int FacIdx(int predIdx) {
    return predIdx >= nPredNum ? predIdx - nPredNum : -1;
  }

  // Returns cardinality of factor at index passed.
  //
  static inline int FacCard(int predIdx) {
    int facIdx = FacIdx(predIdx);
    return facCard[facIdx];
  }

  static inline int MaxFacCard() {
    return maxFacCard;
  }

  // Returns the offset of the factor within the set of factor levels.
  //
  static inline int FacOffset(int predIdx) {
    int facIdx = FacIdx(predIdx);
    return facSum[facIdx];
  }

  static inline int NRow() {
    return nRow;
  }

  static inline int NPred() {
    return nPred;
  }

  static inline int NPredFac() {
    return nPredFac;
  }

  static inline int NPredNum() {
    return nPredNum;
  }

  static inline int NCardTot() {
    return nCardTot;
  }

  // Fixes contiguous factor ordering as numerical preceding factor.
  //
  static inline int PredNumFirst() {
    return 0;
  }
  static inline int PredNumSup() {
    return nPredNum;
  }
  static inline int PredFacFirst() {
    return nPredNum;
  }
  static inline int PredFacSup() {
    return nPred;
  }

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

