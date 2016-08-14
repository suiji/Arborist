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

#include <vector>
#include "param.h"


/**
   @brief Single node type for regression and classification.

   For simplicity, regression and classification variants are distinguished
   only by method name and not by subtyping.  The only distinction is the
   value (and interpretation) of the 'ctg' field.  Care should be taken
   to call the appropriate method, as 'ctg' is only used as a packing
   parameter (with value zero) in the case of regression.  Subtyping seems
   to complicate the code needlessly, with a per-tree size savings of only
   'nSamp' * sizeof(uint).
 */
class SampleNode {
  unsigned int ctg;  // Category of sample; no interpretation for regression.
  FltVal sum; // Sum of values selected:  sCount * y-value.

  // Integer-sized container is likely overkill:  typically << #rows,
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


  inline double Sum() const {
    return sum;
  }
  

  /**
     @brief Accessor for sample count.
     
   */
  inline unsigned int SCount() const {
    return sCount;
  }


};


/**
 @brief Run of instances of a given row obtained from sampling for an individual tree.
*/
class Sample {
  int *row2Sample;
  void PreStage(const class RowRank *rowRank);
  void PreStage(const class RowRank *rowRank, int predIdx);
 protected:
  static unsigned int nRow;
  static unsigned int nPred;
  static int nSamp;
  SampleNode *sampleNode;
  unsigned int bagCount;
  double bagSum;
  class BV *treeBag;
  class SamplePred *samplePred;
  class Bottom *bottom;
  void PreStage(const std::vector<double> &y, const std::vector<unsigned int> &yCtg, const class RowRank *rowRank);

  static unsigned int *RowSample();

 public:
  static class SampleCtg *FactoryCtg(const std::vector<double> &y, const class RowRank *rowRank, const std::vector<unsigned int> &yCtg);
  static class SampleReg *FactoryReg(const std::vector<double> &y, const class RowRank *rowRank, const std::vector<unsigned int> &row2Rank);

  static void Immutables(unsigned int _nRow, unsigned int _nPred, int _nSamp, const double _feSampleWeight[], bool _withRepl, unsigned int _ctgWidth, int _nTree);
  static void DeImmutables();

  Sample();
  void RowInvert(std::vector<unsigned int> &sample2Row) const;
  
  /**
     @brief Accessor for sample count.
   */
  static inline int NSamp() {
    return nSamp;
  }

  
  /**
     @param row row index at which to look up sample index.

     @return Sample index associated with row, or -1 if none.
   */
  inline int SampleIdx(unsigned int row) const {
    return row2Sample[row];
  }
  
  
  /**
     @brief Accessor for bag count.
   */
  inline unsigned int BagCount() const {
    return bagCount;
  }

  
  inline double BagSum() const {
    return bagSum;
  }

  
  inline class Bottom *Bot() {
    return bottom;
  }

  
  inline class SamplePred *SmpPred() {
    return samplePred;
  }

    
  inline const class BV *TreeBag() const {
    return treeBag;
  }


  inline unsigned int SCount(unsigned int sIdx) const {
    return sampleNode[sIdx].SCount();
  }


  inline unsigned int Ref(int sIdx, FltVal &_sum, unsigned int &_sCount) const {
    return sampleNode[sIdx].Ref(_sum, _sCount);
  }

  
  inline FltVal Sum(int sIdx) const {
    return sampleNode[sIdx].Sum();
  }

  
  virtual ~Sample();
};


/**
   @brief Regression-specific methods and members.
*/
class SampleReg : public Sample {
  unsigned int *sample2Rank; // Only client currently leaf-based methods.
  void SetRank(const std::vector<unsigned int> &row2Rank);
 public:
  SampleReg();
  ~SampleReg();

  inline unsigned int Rank(unsigned int sIdx) const {
    return sample2Rank[sIdx];
  }


  void Stage(const std::vector<double> &y, const std::vector<unsigned int> &row2Rank, const class RowRank *rowRank);
};


/**
 @brief Classification-specific sampling.
*/
class SampleCtg : public Sample {
  static unsigned int ctgWidth;
 public:
  SampleCtg();
  ~SampleCtg();
  static void Immutables(unsigned int _ctgWidth, int _nTree);
  static void DeImmutables();

  
  void Stage(const std::vector<unsigned int> &yCtg, const std::vector<double> &y, const class RowRank *rowRank);
};


#endif
