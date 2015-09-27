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

#include "param.h"

class SampleNode {
 public:
  FltVal sum; // Sum of values selected:  sCount * y-value.
  // Integer-sized container is likely overkill.  Size is typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  unsigned int sCount;
};


class SampleNodeCtg : public SampleNode {
 public:
  unsigned int ctg;
  inline unsigned int LevelFields(FltVal &_sum, unsigned int &_sCount) const {
    _sum = sum;
    _sCount = sCount;
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
  double bagSum;
  unsigned int *inBag;
  class SamplePred *samplePred;
  class SplitPred *splitPred;
  int *CountRows();
  void LeafExtent(const int frontierMap[], int leafExtent[]);
 public:
  static void Immutables(unsigned int _nRow, int _nPred, int _nSamp, unsigned int _ctgWidth, int _nTree);
  static void DeImmutables();

  Sample();

  inline int BagCount() {
    return bagCount;
  }

  
  inline double BagSum() {
    return bagSum;
  }

  
  inline class SplitPred *SplPred() {
    return splitPred;
  }

  
  inline class SamplePred *SmpPred() {
    return samplePred;
  }

    
  inline unsigned int *InBag() {
    return inBag;
  }


  void TreeClear();

  
  virtual ~Sample() {
    delete [] inBag;
  }
};


/**
   @brief Regression-specific methods and members.
*/
class SampleReg : public Sample {
  SampleNode *sampleReg;
  unsigned int *sample2Rank; // Only client currently leaf-based methods.
  void Scores(const int frontierMap[], int treeHeight, double score[]);
  int *LeafPos(const int nonTerm[], const int leafExtent[], int treeHeight);
 public:
  SampleReg();
  ~SampleReg();
  static void Immutables();
  void Stage(const double y[], const unsigned int row2Rank[], const class PredOrd *predOrd);
  void Leaves(const int frontierMap[], int treeHeight, int leafExtent[], double score[], const int nonTerm[], unsigned int *rank, unsigned int *sCount);
};


/**
 @brief Categorical-specific sampling.
*/
class SampleCtg : public Sample {
  static unsigned int ctgWidth;
  static double forestScale;  // Jitter scale for forest-wide scores.
  SampleNodeCtg *sampleCtg;
  void Scores(double *leafWeight, int treeHeight, const int nonTerm[], double score[]);
  void LeafWeight(const int frontierMap[], const int nonTerm[], int treeHeight, double *leafWeight);
 public:
  SampleCtg();
  ~SampleCtg();
  static void Immutables(int _ctgWidth, int _nTree);
  void Stage(const int yCtg[], const double y[], const class PredOrd *predOrd);

  void Leaves(const int frontierMap[], int treeHeight, int leafExtent[], double score[], const int nonTerm[], double *leafWeight);
};


#endif
