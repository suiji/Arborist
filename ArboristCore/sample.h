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

/**
   @brief Single node type for regression and classification.

   For simplicity, regression and classification variants are distinguished
   only by method name and not by subtyping.  The only distinction is the
   value (and interpretation) of the 'ctg' field.  Care should be taken
   to call the appropriate method, as 'ctg' is only used as a packing
   parameter (with value zero) in the case of regression.  Subtyping seems
   to complicate the code needlessly, and only results in a size savings of
   #samples * sizeof(uint) per tree.
 */
class SampleNode {
  unsigned int ctg;  // Category of sample; no interpretation for regression.
  FltVal sum; // Sum of values selected:  sCount * y-value.

  // Integer-sized container is likely overkill.  Size is typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  unsigned int sCount;

 public:

  inline void Set(FltVal _sum, unsigned int _sCount, unsigned int _ctg = 0) {
    sum = _sum;
    sCount = _sCount;
    ctg = _ctg;
  }

  /**
     @brief Compound acceessor.

     @param _sum outputs sum.

     @param _sCount outputs sample count.

     @return Category value or default:  classification / regression, plus output reference parameters.
   */
  inline unsigned int Ref(FltVal &_sum, unsigned int &_sCount) const {
    _sum = sum;
    _sCount = sCount;

    return ctg;
  }


  /**
     @brief Accessor for sample count.
     
   */
  inline unsigned int SCount() {
    return sCount;
  }


};


/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/
class Sample {
 protected:
  static unsigned int nRow;
  static unsigned int nPred;
  static int nSamp;
  SampleNode *sampleNode;
  int bagCount;
  double bagSum;
  unsigned int *inBag;
  class SamplePred *samplePred;
  class SplitPred *splitPred;
  int *PreStage(const double y[], const unsigned int *yCtg = 0);
  static int CountRows(int sCountRow[], int sIdxRow[]);
  void LeafExtent(const int frontierMap[], int leafExtent[]);
 public:
  static void Immutables(unsigned int _nRow, unsigned int _nPred, int _nSamp, double _feSampleWeight[], bool _withRepl, unsigned int _ctgWidth, int _nTree);
  static void DeImmutables();

  Sample();

  /**
     @brief Accessor for sample count.
   */
  static inline int NSamp() {
    return nSamp;
  }

  /**
     @brief Accessor for bag count.
   */
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


  inline unsigned int Ref(int sIdx, FltVal &_sum, unsigned int &_sCount) const {
    return sampleNode[sIdx].Ref(_sum, _sCount);
  }

  virtual ~Sample();
};


/**
   @brief Regression-specific methods and members.
*/
class SampleReg : public Sample {
  unsigned int *sample2Rank; // Only client currently leaf-based methods.
  void Scores(const int frontierMap[], int treeHeight, double score[]);
 public:
  SampleReg();
  ~SampleReg();
  static SampleReg *Factory(const double y[], const class RowRank *rowRank, const unsigned int row2Rank[]);
  void Stage(const double y[], const unsigned int row2Rank[], const class RowRank *rowRank);
  void Leaves(const int frontierMap[], int treeHeight, int leafExtent[], double score[], const int nonTerm[], unsigned int *rank, unsigned int *sCount);
  static int *LeafPos(const int nonTerm[], const int leafExtent[], int treeHeight);
};


/**
 @brief Classification-specific sampling.
*/
class SampleCtg : public Sample {
  static unsigned int ctgWidth;
  static double forestScale;  // Jitter scale for forest-wide scores.
  void Scores(double *leafWeight, int treeHeight, const int nonTerm[], double score[]);
  void LeafWeight(const int frontierMap[], const int nonTerm[], int treeHeight, double *leafWeight);
 public:
  SampleCtg();
  ~SampleCtg();
  static SampleCtg *Factory(const double y[], const class RowRank *rowRank, const unsigned int yCtg[]);
  static void Immutables(unsigned int _ctgWidth, int _nTree);
  static void DeImmutables();
  void Stage(const unsigned int yCtg[], const double y[], const class RowRank *rowRank);
  void Leaves(const int frontierMap[], int treeHeight, int leafExtent[], double score[], const int nonTerm[], double *leafWeight);
};


#endif
