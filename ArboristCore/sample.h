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
   @brief Sum / count record for categorical indices.
 */
class SumCount {
  double sum;
  unsigned int sCount;

 public:
  void Init() {
    sum = 0.0;
    sCount = 0;
  }

  inline void Ref(double &_sum, unsigned int &_sCount) const {
    _sum = sum;
    _sCount = sCount;
  }
  
  
  inline void Accum(double _sum, unsigned int _sCount) {
    sum += _sum;
    sCount += _sCount;
  }


  /**
     @brief Subtracts contents of vector passed.
   */
  void Decr(const SumCount &subtrahend) {
    sum -= subtrahend.sum;
    sCount -= subtrahend.sCount;
  }
};


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

  inline double Set(FltVal _yVal, unsigned int _sCount, unsigned int _ctg = 0) {
    sCount = _sCount;
    sum = _yVal * sCount;
    ctg = _ctg;

    return sum;
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
  class BV *treeBag;
  std::vector<unsigned int> row2Sample;

 protected:
  const unsigned int nRow;
  const unsigned int noSample; // Inattainable sample index.
  static unsigned int nSamp;
  std::vector<SampleNode> sampleNode;
  std::vector<SumCount> ctgRoot;
  unsigned int bagCount;
  double bagSum;

  // Factories parametrized by coprocessor state.
  static class SamplePred *SamplePredFactory(const class Coproc *coproc, unsigned int _nPred, unsigned int _bagCount, unsigned int _bufferSize);
  static class SPCtg* SPCtgFactory(const class Coproc *coproc, const class PMTrain *pmTrain, const class RowRank *rowRank, unsigned int bagCount, unsigned int _nCtg);
  static class SPReg* SPRegFactory(const class Coproc *coproc, const class PMTrain *pmTrain, const class RowRank *rowRank, unsigned int bagCount);
  static unsigned int RowSample(std::vector<unsigned int> &sCountRow);
  
  
  void PreStage(const std::vector<double> &y, const std::vector<unsigned int> &yCtg, const class RowRank *rowRank);
  void Stage(const class RowRank *rowRank, class SamplePred *samplePred, class Bottom *bottom, unsigned int predIdx) const;

  virtual double SetNode(unsigned int sIdx, double val, unsigned int sCount, unsigned int ctg) = 0;

 public:
  static class SampleCtg *FactoryCtg(const std::vector<double> &y, const class RowRank *rowRank, const std::vector<unsigned int> &yCtg, unsigned int _nCtg);
  static class SampleReg *FactoryReg(const std::vector<double> &y, const class RowRank *rowRank, const std::vector<unsigned int> &row2Rank);
  virtual class SplitPred *SplitPredFactory(const PMTrain *pmTrain, const RowRank *rowRank, const class Coproc *coproc) const = 0;

  static void Immutables(unsigned int _nSamp, const std::vector<double> &_feSampleWeight, bool _withRepl);
  static void DeImmutables();
  Sample(unsigned int _nRow, unsigned int nCtg);
  virtual ~Sample();

  void Stage(const class RowRank *rowRank, class SamplePred *samplePred, class Bottom *bottom) const;
  class IndexLevel *IndexFactory(const class PMTrain *pmTrain, const class RowRank *rowRank, const class Coproc *coproc) const;


  void RowInvert(std::vector<unsigned int> &sample2Row) const;

  
  /**
     @brief Accessor for sample count.
   */
  static inline unsigned int NSamp() {
    return nSamp;
  }


  /**
     @param row row index at which to look up sample index.

     @return Sample index associated with row, or 'noSample' if none.
   */
  inline bool SampleIdx(unsigned int row, unsigned int &sIdx) const {
    sIdx = row2Sample[row];
    return sIdx != noSample;
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
};


/**
   @brief Regression-specific methods and members.
*/
class SampleReg : public Sample {
  unsigned int *sample2Rank; // Only client currently leaf-based methods.
  void SetRank(const std::vector<unsigned int> &row2Rank);

 public:
  SampleReg(unsigned int _nRow);
  ~SampleReg();
  SplitPred *SplitPredFactory(const PMTrain *pmTrain, const RowRank *rowRank, const class Coproc *coproc) const;

  
  inline double SetNode(unsigned int sIdx, double yVal, unsigned int sCount, unsigned int ctg) {
    SampleNode sNode;
    double ySum = sNode.Set(yVal, sCount);
    sampleNode[sIdx] = sNode;

    return ySum;
  }
  
  
  inline unsigned int Rank(unsigned int sIdx) const {
    return sample2Rank[sIdx];
  }


  void PreStage(const std::vector<double> &y, const std::vector<unsigned int> &row2Rank, const class RowRank *rowRank);
};


/**
 @brief Classification-specific sampling.
*/
class SampleCtg : public Sample {
  const unsigned int nCtg;


 public:
  SampleCtg(unsigned int _nRow, unsigned int _nCtg);
  ~SampleCtg();

  SplitPred *SplitPredFactory(const PMTrain *pmTrain, const RowRank *rowRank, const class Coproc *coproc) const;

  inline double SetNode(unsigned int sIdx, double yVal, unsigned int sCount, unsigned int ctg) {
    SampleNode sNode;
    double ySum = sNode.Set(yVal, sCount, ctg);
    sampleNode[sIdx] = sNode;
    ctgRoot[ctg].Accum(ySum, sCount);

    return ySum;
  }
  
  
  void PreStage(const std::vector<unsigned int> &yCtg, const std::vector<double> &y, const class RowRank *rowRank);
};


#endif
