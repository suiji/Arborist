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

class SampleNode {
 public:
  double sum; // Sum of values selected:  rowRun * y-value.
  // Integer-sized container is likely overkill.  Size is typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  unsigned int rowRun;
};


class SampleNodeCtg : public SampleNode {
 public:
  unsigned int ctg;
  inline unsigned int CtgAndSum(double &_sum) const {
    _sum = sum;
    return ctg;
  }
};

/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/
class Sample {
 protected:
  static int nPred;
  static unsigned int nRow;
  static int nSamp;
  int bagCount;
  int *CountRows();
 public:
  static void Immutables(unsigned int _nRow, int _nPred, int _nSamp);
  static void DeImmutables();


  static inline int NSamp() {
    return nSamp;
  }


  static inline int NRow() {
    return nRow;
  }


  static inline int NPred() {
    return nPred;
  }

  
  inline int BagCount() {
    return bagCount;
  }

  virtual ~Sample() {}
  virtual void Scores(const int frontierMap[], int treeHeight, int leafExtent[], double score[]) = 0;
  virtual int QuantileFields(int sIdx, unsigned int &rank) = 0;
};


/**
   @brief Regression-specific methods and members.
*/
class SampleReg : public Sample {
  SampleNode *sampleReg;
  unsigned int *sample2Rank; // Only client currently quantile regression.

 public:
  SampleReg();
  ~SampleReg();
  static void Immutables();
  int Stage(const double y[], const unsigned int row2Rank[], const class PredOrd *predOrd, unsigned int inBag[], class SamplePred *samplePred, class SplitPred *&splitPred, double &bagSum);
  void Scores(const int frontierMap[], int treeHeight, int leafExtent[], double score[]);
  int QuantileFields(int sIdx, unsigned int &rank);
};


/**
 @brief Categorical-specific sampling.
*/
class SampleCtg : public Sample {
  static int ctgWidth;
  SampleNodeCtg *sampleCtg;
 public:
  SampleCtg();
  ~SampleCtg();
  static void Immutables(int _ctgWidth);
  int Stage(const int yCtg[], const double y[], const class PredOrd *predOrd, unsigned int inBag[], class SamplePred *samplePred, class SplitPred *&splitPred, double &bagSum);
  void Scores(const int frontierMap[], int treeHeight, int leafExtent[], double score[]);
  int QuantileFields(int sIdx, unsigned int &rank);
};


#endif
