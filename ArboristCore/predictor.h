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
  static int *intBase;
  static int *facCount;
  static int *facSum;
  static int *facWidth;
  static bool numClone;  // Whether we keep a private copy of numerical data.
  static bool intClone;  //  ..... integer ...
  static bool facClone;  //  ...... factor .....
 public:
  static int *facBase;   // Ordered factor list.
  static double *predProb;
  static double *numBase;
  static int nRow;
  static int nPred;
  static int nPredInt;
  static int nPredNum;
  static int nPredFac;
  static int facTot; // Total number of levels over all factor predictors.
  static int maxFacWidth;  // Highest number of levels among all factors.
  static void SetProbabilities(const double _predProb[]);
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

  static inline int FacWidth(int facIdx) {
    return facWidth[facIdx];
  }

  // Returns the offset of the factor within the set of factor levels.
  //
  static inline int FacOffset(int facIdx) {
    return facSum[facIdx];
  }

  static inline int NPred() {
    return nPred;
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

  static void SetSortAndTies(const int *rank2Row, Dord *dOrd);
  static void OrderByRank(const int *Col, const int *r2r, Dord *dCol, bool ordinals = true);
  static void OrderByRank(const double *xCol, const int *r2r, Dord *dCol);
  static void Factory(int _nRow);
  static void DeFactory();
};

#endif

