// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sample.h

   @brief Class definitions for sample-oriented aspects of training.

   @author Mark Seligman
 */

#ifndef ARBORIST_SAMPLE_H
#define ARBORIST_SAMPLE_H

/**
   @brief Contains the predictor-specific component of the staged data.
 */
class PredOrd {
 public:
  int rank; // True rank, with ties identically receiving lowest applicable value.
  int row; // local copy of r2r[] value.
};


/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/
class Sample {
  static double *sYRow;
  static void TreeInit();
 protected:
  static int nPred;
  static int nRow;
  static int nSamp;
  static PredOrd *predOrd;
  static int *sCountRow;
  static void CountRows(const int rvRow[]);
 public:
  double sum; // Sum of values selected:  rowRun * y-value.
  // Integer-sized container is likely overkill.  Size is typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  unsigned int rowRun;

  static bool *inBag; // Overwritten by each tree.
  static int *sIdxRow; // Inverted by FacResponse for local use.
  static inline int NSamp() {
    return nSamp;
  }
  static inline int NRow() {
    return nRow;
  }
  static void Factory(int _nRow, int _nPred, int _nSamp);
  static void TreeClear();
  static void DeFactory();
};

/**
   @brief Regression-specific methods and members.
*/
class SampleReg : public Sample {
  static SampleReg *sampleReg;
  static int *sample2Rank;
 public:
  /**
     @brief Computes the sum of all sampled response values.

     @param bagCount is the size of this tree's in-bag set.

     @return sum of sampled values.
  */
  static double Sum(int bagCount) {
    double _sum = 0.0;
    for (int i = 0; i < bagCount; i++)
      _sum += sampleReg[i].sum;

    return _sum;
  }
  static int SampleRows(const int rvRow[],  const double y[], const int row2Rank[]);
  static void Stage();
  static void Stage(int predIdx);
  static void Scores(int bagCount, int leafCount, double score[]);
  static void TreeQuantiles(int treeSize, int bagCount, int leafPos[], int leafExtent[], int rank[], int rankCount[]);
  static void TreeClear();
};

/**
 @brief Categorical-specific sampling.
*/
class SampleCtg : public Sample {
  static SampleCtg *sampleCtg;
 public:
  unsigned int ctg;

  /**
     @brief Computes the sum of sample values for the current tree.

     @param bagCount is the number of samples drawn.

     @return sum of (proxy) response values.
  */
  static double Sum(int bagCount) {
    double _sum = 0.0;
    for (int i = 0; i < bagCount; i++)
      _sum += sampleCtg[i].sum;

    return _sum;
  }

  static int CtgSum(int sIdx, double &_sum) {
    _sum = sampleCtg[sIdx].sum;
    return sampleCtg[sIdx].ctg;
  }

  static int SampleRows(const int rvRow[], const int yCtg[], const double y[]);
  static void Stage();
  static void Stage(int predIdx);
  static void Scores(int bagCount, int ctgWidth, int leafCount, double score[]);
  static void TreeClear();
};


#endif
